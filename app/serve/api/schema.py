from pydantic import BaseModel, Field
from typing import Optional, List 




class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    question: Optional[str] = Field(
        ..., example="Where is London located?", title="Example question."
    )
    context: Optional[str] = Field(
        ..., example="London is in England.", title="Example unhelpful context to aid question"
    )


class InferenceResult(BaseModel):
    """
    Inference result from the model: As previously defied
    """
    answer: Optional[str] = Field(..., example="London is in England, United Kingdom", title="Response to question")


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """

    error: bool = Field(..., example=False, title="Whether there is error")
    result: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """

    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: str = Field(None, example="", title="Detailed traceback of the error")
