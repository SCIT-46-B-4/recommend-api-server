from sqlalchemy import Column, BigInteger, DateTime, Date, ForeignKey, func
from sqlalchemy.orm import relationship

from src.db.orm import Base


class DetailScheduleEntity(Base):
    __tablename__ = "detail_schedules"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    schedule_id = Column(BigInteger, ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)

    date = Column(Date, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    schedule = relationship("ScheduleEntity", back_populates="detail_schedules", lazy="joined")
    routes = relationship("RouteEntity", back_populates="detail_schedule", cascade="all, delete-orphan", lazy="joined")
