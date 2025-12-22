import typing
from pydantic import BaseModel, ValidationError
from kungfu import Result, Ok, Error
from pydantic_core import PydanticSerializationError


def parse[M: BaseModel](
    data: dict[typing.Any, typing.Any], *, model: type[M]
) -> Result[M, ValidationError]:
    try:
        return Ok(model.model_validate(data))
    except ValidationError as e:
        return Error(e)


def dump(
    model: BaseModel,
) -> Result[dict[typing.Any, typing.Any], PydanticSerializationError]:
    try:
        return Ok(model.model_dump())
    except PydanticSerializationError as e:
        return Error(e)
