from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String

class Paper(Base):
    __tablename__ = 'papers'
    embedding = Column(Vector(768))  # For 768-dim embeddings