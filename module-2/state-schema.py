from pydantic import BaseModel, field_validator, ValidationError

class PydanticGraphState(BaseModel):
    name: str
    mood: str

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value):
        if value not in ["happy", "sad"]:
            raise ValueError("Mood must be either happy or sad")
        return value

try:
    state = PydanticGraphState(name="jonathanmv", mood="unacceptable")
except ValidationError as e:
    print("Expected validation error:", e)

# Then you create the graph as usual