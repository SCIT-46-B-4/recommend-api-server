from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Numeric, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from src.db.base import Base


class DestinationEntity(Base):
    __tablename__ = "destinations"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    city_id = Column(Integer, ForeignKey("cities.id", ondelete="SET NULL"), nullable=True)
    # ToDo: Enum으로 교체
    # 1: 관광지, 2: 식당, 3: 쇼핑센터, 4: 숙박업소, 5: 대중교통
    type = Column(String(1), nullable=False)
    kr_name = Column(String(256), nullable=False)
    loc_name = Column(String(256), nullable=False)
    title = Column(String(256), nullable=False)
    content = Column(String(2048), nullable=False)
    latitude = Column(Numeric(10, 7), nullable=True)
    longitude = Column(Numeric(10, 7), nullable=True)
    address = Column(String(512), nullable=True)
    contact = Column(String(128), nullable=True)
    homepage = Column(String(256), nullable=True)
    how_to_go = Column(String(128), nullable=False)
    available_time = Column(String(512), nullable=True)
    feature = Column(JSON, nullable=True)
    score = Column(Numeric(3, 1), nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)
    title_img = Column(String(512), nullable=True)
    # 좌표 정보는 POINT 타입(SRID 4326)으로 저장
    coordinate = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)

    city = relationship("CityEntity", back_populates="destinations", lazy="joined")
    routes = relationship("RouteEntity", back_populates="destination", cascade="all, delete-orphan", lazy="joined")
