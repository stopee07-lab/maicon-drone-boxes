import unittest
from ugv_tracking import UGVTracker


def robot(x=100,y=100):
    return {'center_px':[x,y],'bbox_px':[x-20,y-20,x+20,y+20],'confidence':.8}


class Tracks(unittest.TestCase):
    def test_match_produces_same_track_id(self):
        t=UGVTracker()
        a=t.update([robot()],1,'a',camera='test1')['ugvs'][0]
        b=t.update([robot(110,100)],2,'b',camera='test1')['ugvs'][0]
        self.assertEqual(a['track_id'],b['track_id'])
        self.assertEqual(b['observed_frames'],2)

    def test_legacy_identity_fields_are_excluded_from_track_output(self):
        t=UGVTracker()
        legacy = dict(robot(), team='RED', relation='ALLY', callsign='R-1')
        row=t.update([legacy],1,'a',camera='main')['ugvs'][0]
        self.assertEqual(set(row), {'center_px', 'bbox_px', 'confidence', 'track_id',
                                    'observed_at', 'observed_frames'})

    def test_duplicate_old_and_unknown_timestamps_are_rejected(self):
        t=UGVTracker()
        t.update([robot()],2,'a',camera='main')
        for stamp,key in ((2,'b'),(1,'c'),(3,'a'),(None,'d')):
            self.assertFalse(t.update([robot()],stamp,key,camera='main')['accepted'])

    def test_gap_and_camera_switch_start_new_id(self):
        t=UGVTracker()
        rows=[]
        for stamp,camera in ((1,'main'),(30,'main'),(31,'test1')):
            rows.append(t.update([robot()],stamp,str(stamp),camera=camera)['ugvs'][0]['track_id'])
        self.assertEqual(len(set(rows)),3)

    def test_crossing_ambiguity_does_not_swap_labels(self):
        t=UGVTracker()
        first=t.update([robot(100),robot(140)],1,'a',camera='main')['ugvs']
        current=t.update([robot(118),robot(122)],2,'b',camera='main')['ugvs']
        self.assertFalse({v['track_id'] for v in first} & {v['track_id'] for v in current})

    def test_occluded_robot_not_reported_as_current_detection(self):
        t=UGVTracker()
        t.update([robot()],1,'a',camera='main')
        self.assertEqual(t.update([],2,'b',camera='main')['ugvs'],[])

    def test_tracker_recovers_after_ambiguous_crossing(self):
        t=UGVTracker()
        t.update([robot(100),robot(120)],1,'a',camera='main')
        merged=t.update([robot(110)],2,'b',camera='main')['ugvs'][0]
        for stamp in (3,4,5):
            current=t.update([robot(110)],stamp,str(stamp),camera='main')['ugvs'][0]
            self.assertEqual(current['track_id'],merged['track_id'])
            self.assertGreater(current['observed_frames'],1)


if __name__=='__main__':
    unittest.main()
