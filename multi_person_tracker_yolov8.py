import os
from typing import Dict, List, Optional, Union

import numpy as np
from multi_person_tracker import Sort
from ultralytics import YOLO


class MPT8:
    def __init__(
            self,
            model_type: str = 'yolov8n.pt',
            output_format: str = 'dict',
    ) -> None:
        self.model = YOLO(model_type)
        self.extensions = set(['jpg', 'jpeg', 'png'])
        self.output_format = output_format

    def __call__(self, image_folder: str) -> Optional[Union[Dict, List]]:
        image_paths = sorted(
            [
                os.path.join(image_folder, filename)
                for filename in os.listdir(image_folder)
                if self._check_extension(filename)
            ]
        )
        tracker = Sort()
        trackers = []
        # predictions = self.model(image_paths)  # too much of ram! 6.5GB vs <2GB
        for image_path in image_paths:
            # infere yolo model
            prediction = self.model(image_path)[0].boxes.boxes.cpu().numpy()
            # filter non-human objects, remove class dim
            prediction = prediction[prediction[:, 5] == 0][:, :5]
            # track objects
            if prediction.shape[0] > 0:
                track_trackers = tracker.update(prediction)
            else:
                track_trackers = np.empty((0, 5))
            trackers.append(track_trackers)
        if self.output_format == 'dict':
            result = self._prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers
        else:
            raise ValueError(
                'output_format should be either "dict" or "list", while set '
                f'to {self.output_format}'
            )
        return result

    def _check_extension(self, filename: str) -> bool:
        return filename.split('.')[-1].lower() in self.extensions

    def _prepare_output_tracks(self, trackers):
        '''
        Put results into a dictionary consists of detected people
        :param trackers (ndarray): input tracklets of shape Nx5
            [x1,y1,x2,y2,track_id]
        :return: dict: of people. each key represent single person with
            detected bboxes and frame_ids

        *borrowed from repo: mkocabas/multi-person-tracker
        file: multi-person-tracker/multi_person_tracker/mpt.py
        '''
        people = dict()

        for frame_idx, tracks in enumerate(trackers):
            for d in tracks:
                person_id = int(d[4])

                w, h = d[2] - d[0], d[3] - d[1]
                c_x, c_y = d[0] + w/2, d[1] + h/2
                w = h = np.where(w / h > 1, w, h)
                bbox = np.array([c_x, c_y, w, h])

                if person_id in people.keys():
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
                else:
                    people[person_id] = {
                        'bbox' : [],
                        'frames' : [],
                    }
                    people[person_id]['bbox'].append(bbox)
                    people[person_id]['frames'].append(frame_idx)
        for k in people.keys():
            people[k]['bbox'] = (
                np
                .array(people[k]['bbox'])
                .reshape((len(people[k]['bbox']), 4))
            )
            people[k]['frames'] = np.array(people[k]['frames'])

        return people
