# import datetime
# import time
# from sqlalchemy.orm import Session
# from sqlalchemy.sql import func

# from src.entity.city import CityEntity
# from src.entity.country import CountryEntity
# from src.entity.destination import DestinationEntity
# from src.entity.detail_schedule import DetailScheduleEntity
# from src.entity.route import RouteEntity
# from src.entity.schedule import ScheduleEntity
# from src.dto.response.schedule_response import ScheduleResponse
# from src.repository.base_repository import get_dto_by_id

# # 1: 관광지, 2: 식당, 3: 쇼핑센터, 4: 숙박업소, 5: 대중교통
# def create_schedule(db: Session, user_id: int) -> None:
#     start_time = time.time()
    
#     db.query(ScheduleEntity).delete(synchronize_session=False)
#     db.query(DetailScheduleEntity).delete(synchronize_session=False)
#     db.query(RouteEntity).delete(synchronize_session=False)
#     db.commit()

#     country = db.query(CountryEntity).filter(CountryEntity.kr_name == "일본").first()
#     city = db.query(CityEntity).filter(CityEntity.kr_name == "도쿄").first()
#     city_id: int = city.id

#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"초기세팅: {elapsed} seconds")

#     start_time = time.time()
#     schedule = ScheduleEntity(
#         user_id=user_id,
#         country_id=country.id,
#         city_id=city_id,
#         name=f"{city.kr_name} 여행",
#         start_date=datetime.date.fromisoformat("2024-06-07"),
#         end_date=datetime.date.fromisoformat("2024-06-11"),
#         country_name = country.kr_name,
#         city_name=city.kr_name
#     )
#     db.add(schedule)
#     db.commit()
#     db.refresh(schedule)
#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(f"첫 스케줄 생성: {elapsed} seconds")

#     # for idx, d in enumerate(range(7, 12)):
#     #     start_time = time.time()
#     #     accommodation = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 4).order_by(func.rand()).first()
#     #     asa_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2).order_by(func.rand()).first()
#     #     asa_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3])).order_by(func.rand()).limit(2).all()
#     #     hiru_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id != asa_restaurant.id).order_by(func.rand()).first()
#     #     hiru_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3]), DestinationEntity.id.notin_([a.id for a in asa_attraction])).order_by(func.rand()).limit(2).all()
#     #     ban__restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id.notin_([asa_restaurant.id, hiru_restaurant.id])).order_by(func.rand()).first()
#     #     dest_list = [accommodation, asa_restaurant, asa_attraction, hiru_restaurant, hiru_attraction, ban__restaurant]
#     #     destinations = [item for el in dest_list for item in (el if isinstance(el, list) else [el])]

#     #     detail_date = "2024-06-" + f"0{d}" if d < 10 else "2024-06-" + f"{d}"
#     #     detail_schedule = DetailScheduleEntity(
#     #         schedule_id=schedule.id,
#     #         date=datetime.date.fromisoformat(detail_date)
#     #     )
#     #     db.add(detail_schedule)
#     #     db.commit()
#     #     db.refresh(detail_schedule)
#     #     end_time = time.time()
#     #     elapsed = end_time - start_time
#     #     print(f"{idx+1} 번쨰 디테일 스케줄 생성: {elapsed} seconds")

#     #     start_time = time.time()
#     #     for idx, dest in enumerate(destinations):
#     #         route = RouteEntity(
#     #             detail_schedule_id=detail_schedule.id,
#     #             destination_id=dest.id,
#     #             order_number=idx+1
#     #         )
#     #         db.add(route)
#     # db.commit()
#     # end_time = time.time()
#     # elapsed = end_time - start_time
#     # print(f"라우트 생성 완료: {elapsed} seconds")


#     # schedule = ScheduleEntity(
#     #     user_id=user_id,
#     #     country_id=country.id,
#     #     city_id=city_id,
#     #     name=f"{city.name} 여행",
#     #     start_date=datetime.date.fromisoformat("2024-12-27"),
#     #     end_date=datetime.date.fromisoformat("2024-12-31"),
#     #     country_name = country.name,
#     #     city_name=city.kr_name
#     # )
#     # db.add(schedule)
#     # db.commit()
#     # db.refresh(schedule)

#     # for d in range(27, 32):
#     #     accommodation = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 4).order_by(func.rand()).first()
#     #     asa_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2).order_by(func.rand()).first()
#     #     asa_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3])).order_by(func.rand()).limit(2).all()
#     #     hiru_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id != asa_restaurant.id).order_by(func.rand()).first()
#     #     hiru_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3]), DestinationEntity.id.notin_([a.id for a in asa_attraction])).order_by(func.rand()).limit(2).all()
#     #     ban__restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id.notin_([asa_restaurant.id, hiru_restaurant.id])).order_by(func.rand()).first()
#     #     dest_list = [accommodation, asa_restaurant, asa_attraction, hiru_restaurant, hiru_attraction, ban__restaurant]
#     #     destinations = [item for el in dest_list for item in (el if isinstance(el, list) else [el])]

#     #     detail_date = "2024-12-" + f"0{d}" if d < 10 else "2024-12-" + f"{d}"
#     #     detail_schedule = DetailScheduleEntity(
#     #         schedule_id=schedule.id,
#     #         date=datetime.date.fromisoformat(detail_date)
#     #     )
#     #     db.add(detail_schedule)
#     #     db.commit()
#     #     db.refresh(detail_schedule)

#     #     for idx, dest in enumerate(destinations):
#     #         route = RouteEntity(
#     #             detail_schedule_id=detail_schedule.id,
#     #             destination_id=dest.id,
#     #             order_number=idx+1
#     #         )
#     #         db.add(route)
#     # db.commit()


#     # schedule = ScheduleEntity(
#     #     user_id=user_id,
#     #     country_id=country.id,
#     #     city_id=city_id,
#     #     name=f"{city.name} 여행",
#     #     start_date=datetime.date.fromisoformat("2025-04-08"),
#     #     end_date=datetime.date.fromisoformat("2025-04-12"),
#     #     country_name = country.name,
#     #     city_name=city.kr_name
#     # )
#     # db.add(schedule)
#     # db.commit()
#     # db.refresh(schedule)

#     # for d in range(8, 13):
#     #     accommodation = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 4).order_by(func.rand()).first()
#     #     asa_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2).order_by(func.rand()).first()
#     #     asa_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3])).order_by(func.rand()).limit(2).all()
#     #     hiru_restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id != asa_restaurant.id).order_by(func.rand()).first()
#     #     hiru_attraction = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type.in_([1, 3]), DestinationEntity.id.notin_([a.id for a in asa_attraction])).order_by(func.rand()).limit(2).all()
#     #     ban__restaurant = db.query(DestinationEntity).filter(DestinationEntity.city_id == city_id, DestinationEntity.type == 2, DestinationEntity.id.notin_([asa_restaurant.id, hiru_restaurant.id])).order_by(func.rand()).first()
#     #     dest_list = [accommodation, asa_restaurant, asa_attraction, hiru_restaurant, hiru_attraction, ban__restaurant]
#     #     destinations = [item for el in dest_list for item in (el if isinstance(el, list) else [el])]

#     #     detail_date = "2025-04-" + f"0{d}" if d < 10 else "2025-04-" + f"{d}"
#     #     detail_schedule = DetailScheduleEntity(
#     #         schedule_id=schedule.id,
#     #         date=datetime.date.fromisoformat(detail_date)
#     #     )
#     #     db.add(detail_schedule)
#     #     db.commit()
#     #     db.refresh(detail_schedule)

#     #     for idx, dest in enumerate(destinations):
#     #         route = RouteEntity(
#     #             detail_schedule_id=detail_schedule.id,
#     #             destination_id=dest.id,
#     #             order_number=idx+1
#     #         )
#     #         db.add(route)
#     # db.commit()


# def get_schedule_by_id(db: Session, schedule_id: int) -> ScheduleResponse | None:
#     return get_dto_by_id(db, ScheduleEntity, schedule_id, ScheduleResponse)
