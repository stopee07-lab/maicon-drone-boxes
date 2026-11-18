"""Associate aerial UGV observations by pixel proximity across fresh frames."""
import copy
import math


class UGVTracker:
    """Use fresh source timestamps; report only robots visible in this frame.

    Ambiguous crossings, occlusions and long gaps create new visual track IDs.
    """
    def __init__(self, max_gap_s=12., max_distance_px=90.):
        self.max_gap_s = float(max_gap_s)
        self.max_distance_px = float(max_distance_px)
        self.camera = None
        self.last_stamp = None
        self.last_key = None
        self.tracks = {}
        self.sequence = 0

    def reset(self):
        self.tracks.clear()
        self.last_stamp = self.last_key = None

    def update(self, detections, captured_at, frame_key, *, camera):
        if not isinstance(captured_at, (float,int)) or not math.isfinite(captured_at) or not frame_key:
            return {'accepted':False, 'reason':'source_timestamp_required', 'ugvs':[]}
        if camera != self.camera:
            self.reset()
            self.camera = camera
        if self.last_stamp is not None and (captured_at <= self.last_stamp or frame_key == self.last_key):
            return {'accepted':False, 'reason':'duplicate_or_out_of_order', 'ugvs':[]}
        prior = {key: value for key, value in self.tracks.items()
                 if 0 < captured_at - value['stamp'] <= self.max_gap_s}
        candidates = []
        for item in detections:
            x,y = item['center_px']
            if not all(math.isfinite(float(v)) for v in (x,y)):
                raise ValueError('Non-finite UGV center')
            eligible = []
            for ident,track in prior.items():
                distance = math.hypot(x-track['center'][0], y-track['center'][1])
                gate = self.max_distance_px * min(2., max(.5,captured_at-track['stamp']))
                if distance <= gate:
                    eligible.append((distance,ident))
            eligible.sort()
            # Do not preserve an identity across unresolved close neighbors.
            if len(eligible) > 1 and eligible[1][0]-eligible[0][0] < 15:
                eligible = []
            candidates.append(eligible)
        bests = [items[0][1] if items else None for items in candidates]
        output = []
        next_tracks = {}
        for index,item in enumerate(detections):
            ident = bests[index]
            if ident is not None and bests.count(ident) > 1:
                ident = None
            if ident is None:
                self.sequence += 1
                ident = f'visual-{self.sequence:04d}'
                previous = None
            else:
                previous = prior[ident]
            row = {field: copy.deepcopy(item[field])
                   for field in ('bbox_px', 'center_px', 'confidence', 'detection_evidence')
                   if field in item}
            row.update(track_id=ident,
                       observed_frames=1 if previous is None else previous['frames']+1,
                       observed_at=captured_at)
            output.append(row)
            next_tracks[ident] = {'center':list(row['center_px']), 'stamp':captured_at,
                                  'frames':row['observed_frames']}
        # Retire superseded/occluded identities: they must not perpetually
        # compete with the replacement after an ambiguous crossing.
        self.tracks = next_tracks
        self.last_stamp, self.last_key = captured_at, frame_key
        return {'accepted':True,'reason':'fresh_frame','ugvs':output}
