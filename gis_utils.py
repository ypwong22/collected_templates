############################################################
# ArcGIS code to create buffer around shapefile that has the
# same area as the old
# Modified from https://gis.stackexchange.com/questions/276478/growing-polygons-with-limiting-area-and-boundary-using-arcgis-desktop
############################################################
import arcpy, math
arcpy.env.overwriteOutput = True

#Change to match your data:
arcpy.env.workspace = r'C:MyProject.gdb'
polygons = r'Input'
out_feature_class = r'Out'

#Change to match your desired size of polygon, ok difference and step in buffer radius increase
ok_diff = 1000000
increment = 500

arcpy.CreateFeatureclass_management(out_path=arcpy.env.workspace, out_name=out_feature_class, 
                                    geometry_type='POLYGON', 
                                    spatial_reference=arcpy.Describe(polygons).spatialReference)

arcpy.MakeFeatureLayer_management(in_features=polygons, out_layer='blyr')

with arcpy.da.SearchCursor(polygons,['OID@','SHAPE@']) as cursor:
    for row in cursor:
        bufferarea = row[1].area

        sql = """{0} = {1}""".format(arcpy.AddFieldDelimiters(polygons,arcpy.Describe(polygons).OIDFieldName),row[0])
        arcpy.MakeFeatureLayer_management(in_features=polygons,out_layer='newlyr',where_clause=sql)
        arcpy.SelectLayerByLocation_management(in_layer='blyr', overlap_type='INTERSECT', 
                                               select_features='newlyr')

        if [i[0] for i in arcpy.da.SearchCursor('blyr','SHAPE@AREA')][0] > 7500:
            area = 1
            while abs(bufferarea-area)>ok_diff:
                arcpy.Buffer_analysis(in_features='newlyr', out_feature_class=r'new_polygon', 
                                      buffer_distance_or_field="{0} Meters".format(increment))
                arcpy.Erase_analysis(in_features=r'new_polygon', erase_features='blyr',out_feature_class=r'erasebuffer', cluster_tolerance = 100)
                area = [i[0] for i in arcpy.da.SearchCursor(r'erasebuffer','SHAPE@AREA')][0]
                increment = increment * math.sqrt(bufferarea / area)

                print(bufferarea, area)

            arcpy.Append_management(inputs=r'erasebuffer', target=out_feature_class, schema_type='NO_TEST')
        else:
            print('Impossible to fit buffer inside boundry for polygon number: ',row[0])
