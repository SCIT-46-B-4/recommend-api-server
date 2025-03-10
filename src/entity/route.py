from sqlalchemy import Column, Integer, BigInteger, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from src.db.orm import Base


class RouteEntity(Base):
    __tablename__ = "routes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    detail_schedule_id = Column(BigInteger, ForeignKey("detail_schedules.id", ondelete="CASCADE"), nullable=False)
    destination_id = Column(BigInteger, ForeignKey("destinations.id", ondelete="CASCADE"), nullable=False)

    order_number = Column(Integer, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    destination = relationship("DestinationEntity", back_populates="routes", lazy="joined")
    detail_schedule = relationship("DetailScheduleEntity", back_populates="routes")
