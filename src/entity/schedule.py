from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Date, ForeignKey, func
from sqlalchemy.orm import relationship

from src.db.base import Base


class ScheduleEntity(Base):
    __tablename__ = "schedules"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    country_id = Column(Integer, ForeignKey("countries.id", ondelete="SET NULL"), nullable=True)
    city_id = Column(Integer, ForeignKey("cities.id", ondelete="SET NULL"), nullable=True)

    name = Column(String(32), nullable=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    country_name = Column(String(128), nullable=False)
    city_name = Column(String(128), nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    user = relationship("UserEntity", back_populates="schedules", lazy="joined")
    country = relationship("CountryEntity", back_populates="schedules", lazy="joined")
    city = relationship("CityEntity", back_populates="schedules", lazy="joined")
    detail_schedules = relationship("DetailScheduleEntity", back_populates="schedule", cascade="all, delete-orphan", lazy="joined")
