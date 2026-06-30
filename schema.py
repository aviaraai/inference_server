from fastapi import UploadFile
from pydantic import BaseModel


class Register(BaseModel):
    muzzle_1: UploadFile
    muzzle_2: UploadFile
    muzzle_3: UploadFile
    front_1: UploadFile
    front_2: UploadFile


class Search(BaseModel):
    muzzle: UploadFile
    front: UploadFile
