"""Body geometry regressions; no stored camera footage required."""
import json
import unittest

import cv2
import numpy as np

from ugv_detector import _duplicate, detect_ugvs


def scene():
    image = np.full((720,1280,3),150,np.uint8)
    cv2.rectangle(image,(620,330),(660,390),(25,25,25),-1)
    for y in (340,380):
        cv2.circle(image,(625,y),5,(230,245,230),-1)
    return image


class DetectionGeometry(unittest.TestCase):
    def test_public_output_contains_body_observation_only(self):
        found = detect_ugvs(scene())
        self.assertEqual(len(found),1)
        self.assertEqual(set(found[0]),{'bbox_px','center_px','confidence','detection_evidence'})
        self.assertEqual(set(found[0]['detection_evidence']),
                         {'method','hardware_blob_count','hardware_blobs','complete_census','version'})
        self.assertFalse(found[0]['detection_evidence']['complete_census'])

    def test_blank_and_printed_red_box_icon_are_not_robots(self):
        image = np.full((720,1280,3),150,np.uint8)
        self.assertEqual(detect_ugvs(image),[])
        cv2.rectangle(image,(500,250),(600,350),(30,30,30),-1)
        cv2.rectangle(image,(540,285),(550,310),(250,250,250),-1)
        cv2.rectangle(image,(542,288),(548,307),(30,30,220),-1)
        self.assertEqual(detect_ugvs(image),[])

    def test_dark_printed_box_with_two_white_dots_is_excluded(self):
        image = np.full((720, 1280, 3), 150, np.uint8)
        cv2.rectangle(image, (350, 280), (470, 420), (25, 25, 25), -1)
        for y in (330, 370):
            cv2.circle(image, (390, y), 5, (240, 240, 240), -1)
        self.assertEqual(detect_ugvs(image), [])

    def test_pair_separates_a_chassis_connected_to_dark_border(self):
        image = np.full((720, 1280, 3), 150, np.uint8)
        cv2.rectangle(image, (350, 0), (382, 719), (25, 25, 25), -1)
        cv2.rectangle(image, (380, 320), (420, 380), (25, 25, 25), -1)
        for y in (330, 370):
            cv2.circle(image, (390, y), 5, (240, 240, 240), -1)
        cv2.rectangle(image, (408, 337), (411, 364), (180, 180, 180), -1)
        found = detect_ugvs(image)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['detection_evidence']['method'], 'paired_round_domes_local_chassis')
        x, y = found[0]['center_px']
        self.assertAlmostEqual(x, 401, delta=3)
        self.assertAlmostEqual(y, 350, delta=3)

    def test_floor_contour_cannot_reuse_the_bodys_only_receiver(self):
        anchor = {'x': 50, 'y': 90}
        body = {'box': [30, 40, 80, 100], 'blobs': [anchor, {'x': 50, 'y': 53}]}
        floor = {'box': [25, 87, 85, 145], 'blobs': [anchor]}
        self.assertTrue(_duplicate(floor, body, 1))
        floor['blobs'] = [{'x': 50, 'y': 125}]
        self.assertFalse(_duplicate(floor, body, 1))

    def test_multiple_separated_robots(self):
        image = scene()
        other = scene()
        image[300:410,200:280] = other[300:410,600:680]
        found = detect_ugvs(image)
        self.assertEqual(len(found),2)
        self.assertLess(found[0]['center_px'][0],found[1]['center_px'][0])

    def test_roi_and_original_pixel_coordinates(self):
        image = scene()
        result = detect_ugvs(image,roi=(500,250,800,450))
        self.assertEqual(len(result),1)
        self.assertLess(abs(result[0]['center_px'][0]-640),8)
        self.assertLess(abs(result[0]['center_px'][1]-360),8)
        self.assertEqual(detect_ugvs(image,roi=(0,0,400,720)),[])

    def test_scale_rotation_and_json_serialization(self):
        source = scene()
        for factor,angle in ((.75,0),(1.,90),(1.5,180)):
            h,w = source.shape[:2]
            transform = cv2.getRotationMatrix2D((640,360),angle,1.)
            image = cv2.warpAffine(source,transform,(w,h),borderValue=(150,150,150))
            image = cv2.resize(image,(round(w*factor),round(h*factor)))
            found = detect_ugvs(image)
            self.assertEqual(len(found),1)
            json.dumps(found,allow_nan=False)

    def test_bad_inputs_are_rejected(self):
        for image in (None,np.zeros((20,20),np.uint8),np.zeros((720,1280,3),float)):
            with self.assertRaises(ValueError):
                detect_ugvs(image)
        for roi in ((0,0,2000,720),(3,2,1,10),(0.,0,100,100)):
            with self.assertRaises(ValueError):
                detect_ugvs(scene(),roi)


if __name__ == '__main__':
    unittest.main()
