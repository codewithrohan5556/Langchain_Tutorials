from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name : str
    age : Optional[int] = None
    # email : Optional[EmailStr]
    cgpa : float = Field(gt=0,lt=10,default=5,description='A decimal value representing cgpa of student')
    
new_student = {'name':'rohan',
               'age':'23',
            #    'email':'rohan@gmail.com',
               'cgpa' : '7.8'}
student = Student(**new_student)

print(student)
print(type(student)) 

student_dict = (dict(student))
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)













