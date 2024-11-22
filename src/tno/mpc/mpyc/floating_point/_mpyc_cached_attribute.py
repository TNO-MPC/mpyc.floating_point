"""
This module defines a class that serves as a cached attribute for MPyC SecureObjects.
"""

from __future__ import annotations

from asyncio import Event, Future, Lock
from typing import Callable

from mpyc.finfields import PrimeFieldElement
from mpyc.runtime import Runtime, mpc
from mpyc.sectypes import SecureInteger

from tno.mpc.mpyc.stubs import mpc_coro_ignore


class CachedMPyCAttribute:
    """
    Cached MPyC Attribute serves as a SecureNumber-type attribute that is either initialized or
    computed on-the-fly at the first call. Every subsequent call obtains the cached result.

    Usage:
      - In `SecureFloatingPoint.__init__`, instantiate a CachedMPyCAttribute and assign it to a
        private instance variable. E.g.
            ```
            self._is_zero = CachedMPyCAttribute(
                is_zero_class, lambda: self.significand == 0
            )
            ```
        It is important to name this variable as the argument name to `__init__` prepended with an
        underscore. E.g. if the argument is `is_zero`, then the variable name should be `_is_zero`.
      - If the floating point object is a placeholder, don't do anything more with the attribute.
        Otherwise, make sure to set it through
            ```
            self._is_zero.set(value, attr_name="is_zero")
            ```
        Here, the `attr_name` is used to display a warning message if the provided value is of an
        unexpected type. The argument to `value` can be provided as an integer / SecureInteger /
        FiniteField to set it. Alternatively, if the value is not known, the user should pass
        `value=None`.
        **NOTE: CachedMPyCAttribute.set SHOULD BE CALLED AT SOME POINT, OTHERWISE THE ATTRIBUTE
        CANNOT BE COMPUTED.**
      - Create methods to `SecureFloatingPoint` that provide access to the relevant methods of the
        cached attribute. E.g.
            ```
            def is_zero(self) -> SecureInteger:
                return self._is_zero.compute()
            async def has_cached_is_zero(self) -> bool:
                return await self._is_zero.is_cached()
            ```
      - The SecureFloatingPoint._input function contains some logic that dynamically targets all
        CachedMPyCAttributes and distributes them. Roughly, it does the following:
            ```
            is_attr_cached = await flp.mpc.transfer(
                await secflp._is_zero.is_cached()
                if flp.mpc.pid == sender
                else None,
                senders=sender,
            )
            if is_attr_cached:
                new_is_zero = flp.mpc.input(
                    secflp._is_zero._value, senders=sender
                )
            else:
                new_is_zero = None
            new_secflp = type(secflp)(..., is_zero=new_is_zero)
            ```
      - The reconcile_placeholder function in extended_coroutine.py also contains some logic that
        dynamically targets all CachedMPyCAttributes. Roughly, it does the following:
            ```
            # Let the placeholder know whether the value is computed externally
            reconcile_placeholder_result(
                placeholder._is_zero._value_computed_externally,
                result._is_zero._value_computed_externally,
            )
            # If it is, then provide the value
            if await placeholder_attr._value_computed_externally:
                reconcile_placeholder_result(
                    placeholder._is_zero._value, result._is_zero._value
                )
            ```
    """

    def __init__(
        self,
        rettype: type[SecureInteger],
        compute_func: Callable[[], SecureInteger],
        runtime: Runtime = mpc,
    ) -> None:
        """
        Initialize CachedMPyCAttribute. To actually set the value, please refer to
        `CachedMPyCAttribute.set()`.

        :param rettype: Type of the attribute.
        :param compute_func: Callable that returns the value of the attribute.
        :param runtime: MPyC Runtime that is used by the caller.
        """
        self._rettype = rettype
        self._compute_func = compute_func
        self._rt = runtime

        # The value of the attribute. Usually set through CachedMPyCAttribute.set().
        self._value_internal: SecureInteger = self._rettype(None)

        # Many times, MPyC uses placeholders that are filled eventually. We use
        # _value_computed_externally to await an indicator stating whether _value_pending will be
        # set to an externally available value (e.g. input from another player). The indicator will
        # become available once the other input is instantiated and finished.
        self._value_computed_externally: Future[bool] = self._rt._loop.create_future()

        # Internal indicator that states whether the value is available (either awaited externally
        # or computed internally).
        self._value_computed_event = Event()

        # Indicator whether the value is scheduled for computation. Is set *synchronously* so that
        # a call to `compute` immediately affects calls to `is_cached`, and not after completion of
        # the compute task.
        self._value_scheduled_for_computation = False

        # Lock to ensure that the value is computed only once over possibly many "parallel"
        # computation tasks.
        self._lock = Lock()

    @property
    def _value(self) -> SecureInteger:
        """
        Yield the current value. For internal use only.

        :return: Value of the attribute.
        """
        return self._value_internal

    @_value.setter
    def _value(self, value: SecureInteger) -> None:
        """
        Set value of attribute.

        :param value: Value to set.
        """
        self._value_internal.set_share(value.share)

    def set(
        self, value: int | SecureInteger | PrimeFieldElement | None, attr_name: str = ""
    ) -> None:
        """
        Set the attribute to the provided value. `value=None` indicates that the attribute will not
        be set externally and allows the CachedMPyCAttribute to compute the attribute.

        This method converts the attribute to the expected secure type if needed.

        :param value: Value to set the attribute to. If None, the attribute is not set and the
            CachedMPyCAttribute object is allowed to compute the attribute.
        :param attr_name: Name of the attribute to use in an error message (invalid type).
        """
        if value is None:
            self._set_value_computed_externally(False)
            return
        if isinstance(value, int):
            self._value = self._rettype(value)
        else:
            self._value = convert_to_stype(value, self._rettype, attr_name)
        self._set_value_computed_externally(True)

    def _set_value_computed_externally(self, is_available: bool) -> None:
        """
        Indicate whether the value of this CachedMPyCAttribute is (or will be) externally
        available.

        The value of the CachedMPyCAttribute can be computed upon first call, or can be provided
        by an external source (initializer, another task, ...). Whether it is provided elsewhere
        is often known only at a later time. The CachedMPyCAttribute will not perform a computation
        until it knows for sure that the value is not provided from an external source. Calling
        this function indicates whether the value is (or will be) provided from another source
        releases the lock on the computation function.

        If True, the CachedMPyCAttribute will await the value that is set to the `_value` attribute.
        Note that the external value needs to be set through
        cached_mpyc_attribute._value = ...

        If False, the CachedMPyCAttribute is allowed to compute the value itself, if required.

        :param is_available: Boolean indicating whether the result is (or will be) provided
            externally.
        """
        self._value_computed_externally.set_result(is_available)

    def compute(self) -> SecureInteger:
        """
        Compute the attribute if needed and return the (cached) result.

        :return: Result of the configured computation.
        """
        self._value_scheduled_for_computation = True

        @mpc_coro_ignore
        async def cache_managing_compute(
            self: CachedMPyCAttribute,
        ) -> SecureInteger:
            """
            Compute provided function and (asynchronously) cache the result.

            :return: Result of the configured computation.
            """
            await self._rt.returnType(self._rettype)

            # If the attribute is computed externally, we just need to await that
            if await self._value_computed_externally:
                self._value_computed_event.set()

            # Compute the attribute if it is not already being computed in another task
            if (
                not self._value_computed_event.is_set() and not self._lock.locked()
            ):  # Avoid locking when possible
                async with self._lock:
                    if (
                        not self._value_computed_event.is_set()
                    ):  # if two tasks managed to get here, avoid duplicate computation
                        self._value = self._compute_func()
                        self._value_computed_event.set()

            # Await completion of computation (by another task)
            await self._value_computed_event.wait()
            return self._value

        return cache_managing_compute(self)

    async def is_cached(self) -> bool:
        """
        Indicate whether the attribute is (scheduled to be) cached.

        :return: True if the attribute is (scheduled to be) cached, False otherwise.
        """
        return (
            await self._value_computed_externally
            or self._value_scheduled_for_computation
        )


def convert_to_stype(
    value: int | PrimeFieldElement | SecureInteger,
    stype: type[SecureInteger],
    name: str,
) -> SecureInteger:
    """
    Convert a provided value into the provided secure integer type.

    :param value: Value to be converted.
    :param stype: Type to be converted into.
    :param name: Name of the parameter that this value represents. Used to specify the error
        message.
    :raise TypeError: Provided value cannot be converted.
    :return: Value that is converted into the requested type.
    """
    if isinstance(value, stype):
        return value
    if isinstance(value, (int, stype.field)):
        return stype(value)
    raise TypeError(
        f"Expected {name} of type int, {stype} or {stype.field}, but received {type(value)}."
    )
