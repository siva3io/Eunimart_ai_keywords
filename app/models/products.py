from .db_base import MySqlBase
from sqlalchemy import Column, String


class Products(MySqlBase):
    __tablename__           = 'products'
    id                      = Column(String, primary_key=True)
    fid                     = Column(String)
    image_2                 = Column(String)
    product_title           = Column(String)
    description             = Column(String)

class NewProducts(MySqlBase):
    __tablename__           = 'new_products'
    id                      = Column(String, primary_key=True)
    fid                     = Column(String)
    image_2                 = Column(String)
    product_title           = Column(String)
    description             = Column(String)