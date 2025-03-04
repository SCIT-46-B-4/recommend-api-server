from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship

from src.db.base import Base


class CountryEntity(Base):
    __tablename__ = "countries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    kr_name = Column(String(64), nullable=True)
    eng_name = Column(String(64), nullable=True)
    iso3 = Column(String(3), nullable=True)
    iso2 = Column(String(3), nullable=True)
    continent = Column(String(64), nullable=True)
    continent_code = Column(String(3), nullable=True)
    currency_code = Column(String(3), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    cities = relationship("CityEntity", back_populates="country", cascade="all, delete-orphan", lazy="joined")
    schedules = relationship("ScheduleEntity", back_populates="country", cascade="all, delete-orphan", lazy="joined")
