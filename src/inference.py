import os
import sys
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

class MunicipalReporter:
    def __init__(self, weights_path='models/best.pt'):
        self.model = YOLO(weights_path)
        self.output_csv = 'outputs/csv/municipal_road_report.csv'

    def get_gps_data(self, image_path):
        """Extracts GPS coordinates or returns 'DATA_NOT_FOUND'."""
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if not exif_data:
                return "DATA_NOT_FOUND", "DATA_NOT_FOUND", "MISSING"

            gps_info = {}
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_info[sub_tag] = value[t]

            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                def to_decimal(coords, ref):
                    d, m, s = [float(x) for x in coords]
                    decimal = d + (m / 60.0) + (s / 3600.0)
                    if ref in ['S', 'W']: decimal = -decimal
                    return round(decimal, 6)

                lat = to_decimal(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
                lon = to_decimal(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
                return lat, lon, "VALID"
        except Exception:
            pass
        return "DATA_NOT_FOUND", "DATA_NOT_FOUND", "MISSING"

    def run_inference(self, source_path):
        results = self.model.predict(source=source_path, conf=0.25, imgsz=640)
        report_data = []

        for r in results:
            img_name = os.path.basename(r.path)
            lat, lon, status = self.get_gps_data(r.path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            counts = {'Pothole': 0, 'Crack': 0}
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls)]
                if 'Pothole' in cls_name:
                    counts['Pothole'] += 1
                else:
                    counts['Crack'] += 1

            report_data.append({
                'File_Name': img_name,
                'Latitude': lat,
                'Longitude': lon,
                'Metadata_Status': status,  # Explicitly flags missing data
                'Total_Issues': counts['Pothole'] + counts['Crack'],
                'Pothole_Count': counts['Pothole'],
                'Crack_Count': counts['Crack'],
                'Urgency_Level': "HIGH" if counts['Pothole'] > 0 else "LOW",
                'Timestamp': timestamp
            })
            r.save(filename=os.path.join('outputs/samples', f"report_{img_name}"))

        df = pd.DataFrame(report_data)
        df.to_csv(self.output_csv, index=False)
        print(f"✅ Report Generated. Metadata Status: {df['Metadata_Status'].value_counts().to_dict()}")

if __name__ == "__main__":
    reporter = MunicipalReporter()
    target = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    reporter.run_inference(target)
