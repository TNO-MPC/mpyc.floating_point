# pylint: disable=invalid-name
# We redefine mpyc methods and want to be consistent with their use of CamelCase.
"""
Module that extends the coroutine module in MPyC to support Secure Floating Point objects
"""
from __future__ import annotations

import functools
import sys
from asyncio import Future, Task
from collections.abc import Coroutine
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from mpyc.asyncoro import _Awaitable, _ncopy, _ProgramCounterWrapper, _wrap_in_coro
from mpyc.finfields import PrimeFieldElement
from mpyc.sectypes import SecureNumber

from tno.mpc.mpyc.stubs import returnType as mpyc_stubs_returnType

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point._mpyc_cached_attribute import CachedMPyCAttribute

if TYPE_CHECKING:
    from tno.mpc.mpyc.floating_point.secure_floating_point import SecureFloatingPoint

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

PYTHON_GE_39 = sys.version_info >= (3, 9)  # asyncio.Future is Generic since Python 3.9

T = TypeVar("T")
SecureNumberT = TypeVar("SecureNumberT", bound=SecureNumber)
SecureNumberT2 = TypeVar("SecureNumberT2", bound=SecureNumber)
SecureNumberT3 = TypeVar("SecureNumberT3", bound=SecureNumber)
SecureNumberT4 = TypeVar("SecureNumberT4", bound=SecureNumber)
P = ParamSpec("P")


# region Typing overloads


@overload
def returnType(
    return_type: type[T],
    *,
    wrap: Literal[True] = ...,
) -> _Awaitable[T]: ...


@overload
def returnType(
    return_type: type[T],
    *,
    wrap: Literal[False],
) -> T: ...


@overload
def returnType(
    return_type: type[T],
    *dimensions: int,
    wrap: Literal[True] = ...,
) -> _Awaitable[T] | _Awaitable[list[T]] | _Awaitable[list[list[T]]]: ...


@overload
def returnType(
    return_type: type[T],
    *dimensions: int,
    wrap: Literal[False],
) -> T | list[T] | list[list[T]]: ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], bool],
    *dimensions: int,
    wrap: Literal[True] = ...,
) -> (
    _Awaitable[SecureNumberT]
    | _Awaitable[list[SecureNumberT]]
    | _Awaitable[list[list[SecureNumberT]]]
): ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], bool],
    *dimensions: int,
    wrap: Literal[False],
) -> SecureNumberT | list[SecureNumberT] | list[list[SecureNumberT]]: ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], type[SecureNumberT2]],
    *,
    wrap: Literal[True] = ...,
) -> _Awaitable[tuple[SecureNumberT, SecureNumberT2]]: ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], type[SecureNumberT2]],
    *,
    wrap: Literal[False],
) -> tuple[SecureNumberT, SecureNumberT2]: ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], type[SecureNumberT2], type[SecureNumberT3]],
    *,
    wrap: Literal[True] = ...,
) -> _Awaitable[tuple[SecureNumberT, SecureNumberT2, SecureNumberT3],]: ...


@overload
def returnType(
    return_type: tuple[type[SecureNumberT], type[SecureNumberT2], type[SecureNumberT3]],
    *,
    wrap: Literal[False],
) -> tuple[SecureNumberT, SecureNumberT2, SecureNumberT3]: ...


@overload
def returnType(
    return_type: tuple[
        type[SecureNumberT],
        type[SecureNumberT2],
        type[SecureNumberT3],
        type[SecureNumberT4],
    ],
    *,
    wrap: Literal[True] = ...,
) -> _Awaitable[
    tuple[SecureNumberT, SecureNumberT2, SecureNumberT3, SecureNumberT4],
]: ...


@overload
def returnType(
    return_type: tuple[
        type[SecureNumberT],
        type[SecureNumberT2],
        type[SecureNumberT3],
        type[SecureNumberT4],
    ],
    *,
    wrap: Literal[False],
) -> tuple[SecureNumberT, SecureNumberT2, SecureNumberT3, SecureNumberT4]: ...


# endregion


@functools.wraps(mpyc_stubs_returnType)
def returnType(
    return_type: Any,
    *dimensions: int,
    wrap: bool = True,
) -> Any:
    def returnType_no_wrap(return_type, *dimensions: int):  # type: ignore # same type annotation as returnType, but without the YieldAwaitable wrapper
        # tuple of secure types
        if isinstance(return_type, tuple) and all(
            isinstance(_, type) and issubclass(_, SecureNumber) for _ in return_type
        ):
            if dimensions:
                raise ValueError(
                    "Does not expect argument 'dimensions' if return_type is a tuple of SecureNumber subclasses."
                )
            return tuple(returnType_no_wrap(secure_type) for secure_type in return_type)
        # single secure type
        return mpyc_stubs_returnType(return_type, *dimensions, wrap=False)

    rettype = returnType_no_wrap(return_type, *dimensions)
    return _Awaitable(rettype) if wrap else rettype


def reconcile_placeholder_task(
    placeholder: SecureNumberT, task: Task[SecureNumberT]
) -> None:
    """
    Wait until the task has finished and then set the results to the correct objects.

    :param placeholder: A Dummy variable that needs to be updated with the tasks's result.
    :param task: The task that is running computations.
    :raises Exception: when an exception is raised inside the task.
    """
    flp.mpc._pc_level -= 1
    if placeholder is None:
        return

    try:
        task_result = task.result()
    except Exception:
        flp.mpc._loop.stop()  # pylint: disable=protected-access
        raise

    reconcile_placeholder_result(placeholder, task_result)


def reconcile_placeholder_result(
    placeholder: (
        SecureNumber
        | Future[bool]
        | Future[SecureNumber]
        | list[SecureNumber | Future[SecureNumber]]
        | tuple[SecureNumber | Future[SecureNumber], ...]
    ),
    result: (
        SecureNumber
        | Future[bool]
        | Future[PrimeFieldElement]
        | list[SecureNumber | Future[PrimeFieldElement]]
        | tuple[SecureNumber | Future[PrimeFieldElement], ...]
    ),
) -> None:
    """
    Update a dummy variable with the result of a task.

    :param placeholder: Dummy variable that needs to be updated with the result.
    :param result: A SecureObject that was the result of a Task.
    :raises TypeError: If the types of placeholder and result are inconsistent.
    :raises ValueError: If the placeholder is already set
    """
    # This import needs to go inside this function to avoid circular imports
    # pylint: disable=import-outside-toplevel
    from tno.mpc.mpyc.floating_point import SecureFloatingPoint

    # pylint: enable=import-outside-toplevel
    if isinstance(placeholder, (list, tuple)):
        if not isinstance(result, (list, tuple)):
            raise TypeError(
                f"Expect result to be a list or tuple, but it is {type(result)}"
            )
        for element_placeholder, element_result in zip(placeholder, result):
            reconcile_placeholder_result(element_placeholder, element_result)
        return

    if isinstance(result, (list, tuple)):
        raise TypeError(
            f"result type is incompatible with placeholder type. result type = {type(result)}, "
            f"placeholder type = {type(placeholder)}"
        )

    if isinstance(placeholder, Future):
        if isinstance(result, Future):
            # do not wait for a future to complete in asynchronous mode
            if not flp.mpc.options.no_async:
                if TYPE_CHECKING or PYTHON_GE_39:
                    result = cast(Future[bool], result)
                    result.add_done_callback(
                        lambda fut: cast(Future[bool], placeholder).set_result(
                            fut.result()
                        )
                    )
                else:
                    result.add_done_callback(
                        lambda fut: placeholder.set_result(fut.result())
                    )
            else:
                if TYPE_CHECKING or PYTHON_GE_39:
                    result = cast(Future[bool], result)
                    placeholder = cast(Future[bool], placeholder)
                placeholder.set_result(result.result())
        else:
            if TYPE_CHECKING:
                placeholder = cast(Future[SecureNumber], placeholder)
            placeholder.set_result(result)
        return

    # --- placeholder is a SecureNumber ---
    if isinstance(placeholder, SecureFloatingPoint):
        if not isinstance(result, SecureFloatingPoint):
            raise TypeError(
                "Expected a SecureFloatingPoint to set a result, but received object of type "
                f"{type(result)}."
            )
        reconcile_placeholder_result(placeholder.significand, result.significand)
        reconcile_placeholder_result(placeholder.exponent, result.exponent)
        # pylint: disable=protected-access

        async def reconcile_cached_mpyc_attribute(
            placeholder_attr: CachedMPyCAttribute, result_attr: CachedMPyCAttribute
        ) -> None:
            """
            Update a dummy CachedMPyCAttribute with the final result.

            :param placeholder_attr: Dummy CachedMPyCAttribute that needs to be updated with the
                result.
            :param result_attr: A SecureObject that was the result of a Task.
            """
            # Let the placeholder know whether the value is computed externally
            reconcile_placeholder_result(
                placeholder_attr._value_computed_externally,
                result_attr._value_computed_externally,
            )
            # If it is, then provide the value
            if await placeholder_attr._value_computed_externally:
                reconcile_placeholder_result(
                    placeholder_attr._value, result_attr._value
                )

        cached_mpyc_attribute_names = [
            attr
            for attr in dir(result)
            if isinstance(getattr(result, attr), CachedMPyCAttribute)
        ]
        for cached_attr in cached_mpyc_attribute_names:
            flp.mpc._loop.create_task(
                reconcile_cached_mpyc_attribute(
                    getattr(placeholder, cached_attr), getattr(result, cached_attr)
                )
            )
        return

    if isinstance(result, SecureFloatingPoint):
        raise TypeError(
            f"Expected a SecureObject to set a result, but received object of type {type(result)}."
        )

    if not isinstance(placeholder.share, Future):
        raise ValueError("Trying to reconcile a share that is already set.")

    if isinstance(result, SecureNumber):
        result_value_obj = result.share
    else:
        if TYPE_CHECKING:
            result = cast(Future[PrimeFieldElement], result)
        result_value_obj = result

    # do not wait for a future to complete in asynchronous mode
    if isinstance(result_value_obj, Future) and not flp.mpc.options.no_async:
        set_placeholder_result = (
            placeholder.share.set_result
        )  # lambda functions evaluate lazily, which throws mypy off
        result_value_obj.add_done_callback(lambda x: set_placeholder_result(x.result()))
        return

    # set value (wait for completion if necessary)
    if isinstance(result_value_obj, Future):
        placeholder.share.set_result(result_value_obj.result())
    else:
        placeholder.share.set_result(result_value_obj)  # type: ignore
        # the share will always be a PrimeFieldElement, as SecureFloatingPoints have already been processed earlier


def mpc_coro_generic(
    reconciliation_function_task: Callable[[SecureNumberT, Task[SecureNumberT]], None],
    reconciliation_function_result: Callable[[SecureNumberT, SecureNumberT], None],
    func: Callable[P, Coroutine[Any, Any, SecureNumberT]],
    apply_program_counter_wrapper: bool = True,
    ignore_type_hints: bool = False,
) -> Callable[P, SecureNumberT]:
    """Decorator turning coroutine func into an MPyC coroutine.
    An MPyC coroutine is evaluated asynchronously, returning empty placeholders.
    The type of the placeholders is defined either by a return annotation
    of the form "-> expression" or by the first await expression in func.
    Return annotations can only be used for static types.

    :param reconciliation_function_task: function that takes a task and assigns its result to a
        dummy value
    :param reconciliation_function_result: function that assigns a result to a dummy value
    :param func: The async function to be wrapped
    :param apply_program_counter_wrapper: A boolean value indicating whether a program counter
        wrapper should be applied
    :param ignore_type_hints: A boolean indicating whether type annotations should be used by the
        code to deduce the type of the placeholder
    :return: A new sync function that returns a placeholder for which a result will automatically be set when the
        coroutine has finished running
    """
    rettype = None if ignore_type_hints else get_type_hints(func).get("return")

    @functools.wraps(func)
    def typed_asyncoro(*args: P.args, **kwargs: P.kwargs) -> SecureNumberT:
        """
        This is the function that is returned when the mpc_coro wrapper is applied to an
        async function. This function creates the async coroutine that was wrapped using the
        positional arguments and keyword arguments and assigns the coroutine to a Task. A place-
        holder of the correct type is returned by this function and the value of the placeholder is
        substituted for the actual result when the Task has finished running the coroutine.

        :param args: positional arguments for the async function being wrapped
        :param kwargs: keyword arguments for the async function being wrapped
        :return: A placeholder of the right return type
        :raise Exception: This occurs when either the coroutine does not call returnType or another
            exception is raised while trying to retrieve the right return type.
        """
        flp.mpc._pc_level += 1
        coro = func(*args, **kwargs)
        placeholder: SecureNumberT
        if rettype:
            return_type = cast(type[SecureNumberT], rettype)
            placeholder = returnType(return_type, wrap=False)
        else:
            try:
                # attempting to reach an await returnType(...) statement
                placeholder = coro.send(None)
            except StopIteration as exc:
                # the coroutine returned a value, no returnType encountered
                # the value is not the placeholder but the actual result
                flp.mpc._pc_level -= 1
                return_value: SecureNumberT = exc.value
                return return_value

            except Exception:
                flp.mpc._pc_level -= 1
                raise

        # if this should not be done asynchronously, we exhaust the generator until we get a result
        if flp.mpc.options.no_async:
            while True:
                try:
                    coro.send(None)
                except StopIteration as exc:
                    flp.mpc._pc_level -= 1
                    if placeholder is not None:
                        reconciliation_function_result(placeholder, exc.value)
                    return placeholder

                except Exception:
                    flp.mpc._pc_level -= 1  # pylint: disable=W0212
                    raise

        # we start a new Task that runs the coroutine and instruct it to replace the placeholder
        # when the coroutine has finished
        if apply_program_counter_wrapper:
            coro = _wrap_in_coro(
                _ProgramCounterWrapper(flp.mpc, coro)
            )  # pylint: disable=W0212
        # start the coroutine in a different task
        task = Task(coro, loop=flp.mpc._loop)  # pylint: disable=protected-access
        # enclosing MPyC coroutine call
        # noinspection PyUnresolvedReferences
        # the method is protected, but we do need it, so the inspection tools will throw an error
        # pylint: disable=invalid-name protected-access
        task.f_back = sys._getframe(1)  # type: ignore
        # pylint: enable=invalid-name protected-access

        # make sure the placeholder is replaced after the coroutine is finished
        task.add_done_callback(lambda t: reconciliation_function_task(placeholder, t))
        placeholder_copy = _ncopy(placeholder)
        return placeholder_copy

    return typed_asyncoro


def mpc_coro_extended(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """
    A wrapper for an MPC coroutine that ensures that the behaviour of the code is unaffected by
    the type annotations and supports secure floating points.

    :param func: The async function to be wrapped
    :return: A new sync function that returns a placeholder for which a result will automatically be set when the
        coroutine has finished running
    """
    return mpc_coro_generic(  # type: ignore # mypy cannot infer type argument
        reconciliation_function_task=reconcile_placeholder_task,  # type: ignore[arg-type]
        reconciliation_function_result=reconcile_placeholder_result,  # type: ignore[arg-type]
        func=func,
        apply_program_counter_wrapper=True,
        ignore_type_hints=True,
    )
