
class GeomCollection:
    def __init__(self, session: Session):
        """
        :param session: A session built with :func:`geomesh.db.orm.get_session`
        """
        self.session = session

    def add(self, geom: Geom) -> None:

        if not geom.crs.equals(CRS.from_epsg(4326)):
            logger.info(f"Transforming from {geom.crs} to EPSG:4326 for database storage.")
            geom = Geom(geom.get_multipolygon(crs=CRS.from_epsg(4326)))

        self.session.add(
            db.orm.GeomCollection(
                geom=geoalchemy2.shape.from_shape(geom.multipolygon),
                id=geom.md5,
                )
            )
        self.session.commit()

    def get(self, id: str):
        query = self.session.query(db.orm.Geom).get(id)
        if query is None:
            return
        geom = Geom(
            geoalchemy2.shape.to_shape(query.geom),
            crs=EPSG_4326
        )
        original_crs = CRS.from_user_input(query.original_crs)
        if original_crs.equals(EPSG_4326) is not True:
            geom = Geom(
                    geom.get_multipolygon(crs=original_crs),
                    crs=original_crs
                )
        return geom
