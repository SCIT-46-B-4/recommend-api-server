from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from src.db.base import Base


class CityEntity(Base):
    __tablename__ = "cities"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    country_id = Column(Integer, ForeignKey("countries.id", ondelete="SET NULL"))
    kr_name = Column(String(64), nullable=True)
    eng_name = Column(String(64), nullable=True)
    city_code = Column(String(3), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    country = relationship("CountryEntity", back_populates="cities", lazy="joined")
    destinations = relationship("DestinationEntity", back_populates="city", passive_deletes=True, lazy="joined")
    schedules = relationship("ScheduleEntity", back_populates="city", passive_deletes=True, lazy="joined")
