from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# ======= Pydantic model và validation =======

class Address(BaseModel):
    street: str = Field(..., min_length=1, max_length=100, description="Street address (max 100 characters)", examples=["123 Main St", "456 Elm St"])
    city: str = Field(..., min_length=1, max_length=50, description="City name (max 50 characters)", examples=["New York", "Los Angeles"])

class PatientData(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="Patient's name (max 50 characters)", examples=["John Doe", "Jane Smith"])
    age: int = Field(..., gt=0, lt=120, description="Patient's age, must be greater than 0", examples=[25, 30, 45])
    email: str | None = Field(default=None, description="Patient's email", examples=["john.doe@example.com", "jane.smith@outlook.com"])
    weight: float = Field(..., gt=0, description="Patient's weight in kg, must be greater than 0", examples=[70.5, 69.0])
    height: float = Field(gt=0, description="Patient's height in cm, must be greater than 0", examples=[175.0, 160.5])
    allergies: list[str] | None = Field(default=None, max_length=5, description="Patient's allergies", examples=["penicillin", "peanuts"])
    contact_info: dict[str, str] = Field(..., description="Patient's contact information", examples=[{"phone": "123-456-7890"}, {"email": "john.doe@example.com"}])
    address: Address = Field(..., description="Patient's address", examples=[{"street": "123 Main St", "city": "New York"}])

    @field_validator('email')
    @classmethod
    def email_validator(cls, v):
        valid_domain = ["gmail.com", "yahoo.com", "outlook.com", "example.com", "test.com"]
        if v is None:
            return v
        domain = v.split('@')[-1]
        if domain not in valid_domain:
            raise ValueError(f"Email domain must be one of {valid_domain}")
        return v
    
    @model_validator(mode='after')
    def validate_emergency_contact(self):
        if self.age > 60 and not self.contact_info:
            raise ValueError("Patients over 60 must have an emergency contact")
        return self
    
    @computed_field
    @property
    def bmi(self) -> float:
        height_m = self.height / 100  # Convert cm to meters
        return round(self.weight / (height_m ** 2), 2)


def add_patient_data(patient_data: PatientData):
    print(f"patient name: {patient_data.name}")
    print(f"patient age: {patient_data.age}")
    print(f"patient weight: {patient_data.weight}")
    print(f"patient height: {patient_data.height}")
    print(f"patient allergies: {patient_data.allergies}")
    print(f"patient contact info: {patient_data.contact_info}")
    print(f"patient BMI: {patient_data.bmi}")
    print("Add patient data successfully in database")


def update_patient_data(patient_data: PatientData):
    print(f"patient name: {patient_data.name}")
    print(f"patient age: {patient_data.age}")
    print(f"patient weight: {patient_data.weight}")
    print(f"patient height: {patient_data.height}")
    print(f"patient allergies: {patient_data.allergies}")
    print(f"patient contact info: {patient_data.contact_info}")
    print(f"patient BMI: {patient_data.bmi}")
    print("Update patient data successfully in database")


patient_address_data = {
    "street": "123 Main St",
    "city": "New York"
}

patients_data = {
    "name": "Hihihi",
    "age": 30,
    "weight": 70.5,
    "height": 175.0,
    "allergies": ["penicillin", "peanuts"],
    "contact_info": {"phone": "123-456-7890"},
    "address": patient_address_data
}


def main():
    add_patient_data(PatientData(**patients_data))

if __name__ == "__main__":
    main()