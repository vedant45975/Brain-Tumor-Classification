from pydantic import BaseModel, EmailStr, field_validator

class Register(BaseModel):
    hospital_name: str
    email: EmailStr
    contact: str
    name: str
    address: str
    username: str
    password: str
    
    @field_validator('password')
    @classmethod
    def truncate_password(cls, v):
        # Bcrypt has a 72-byte limit, truncate password if necessary
        password_bytes = v.encode('utf-8')
        if len(password_bytes) > 72:
            v = password_bytes[:72].decode('utf-8', errors='ignore')
        return v

class Login(BaseModel):
    username: str
    password: str
