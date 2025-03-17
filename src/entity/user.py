from sqlalchemy import Column, BigInteger, String, DateTime, Boolean, func
from sqlalchemy.orm import relationship

from src.db.orm import Base


class UserEntity(Base):
    __tablename__ = "users"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    nickname = Column(String(32), nullable=False)
    email = Column(String(256), nullable=False)
    password = Column(String(128), nullable=True)
    phone = Column(String(64), nullable=False)
    is_agree_loc = Column(Boolean, nullable=False, default=False)
    is_agree_news_noti = Column(Boolean, nullable=False, default=False)
    is_agree_marketing_noti = Column(Boolean, nullable=False, default=False)
    join_date = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    profile_img = Column(String(512), nullable=True, default="default_profile_img_url")

    schedules = relationship("ScheduleEntity", back_populates="user", cascade="all, delete-orphan")
