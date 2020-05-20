from SiamMask_master.experiments.siammask_sharp.custom import Custom
from SiamMask_master.tools.test import siamese_track, siamese_init
from SiamMask_master.utils.load_helper import load_pretrain
from SiamMask_master.utils.config_helper import load_config
import torch.utils.data
import dtlpy as dl
import numpy as np
import logging
import torch
import time
import cv2
import os

logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Service runner class

    """

    def __init__(self, project_name=None, package_name=None):
        # ini params
        self.device = None
        self.cfg = None
        self.siammask = None
        self.weights_path = 'weights'
        self.inputs_path = 'inputs'

        if not os.path.isdir(self.inputs_path):
            os.makedirs(self.inputs_path)

        project = dl.projects.get(project_name=project_name)

        project.artifacts.download(local_path=self.weights_path,
                                   package_name=package_name,
                                   artifact_name='config_davis.json')

        project.artifacts.download(local_path=self.weights_path,
                                   package_name=package_name,
                                   artifact_name='SiamMask_DAVIS.pth')

        # init models
        self.init_siam_model()
        self.cache = dict()
        self.cache_max_size = 5

    def init_siam_model(self):
        class TempLoader:
            resume = os.path.join(self.weights_path, 'SiamMask_DAVIS.pth')
            config = os.path.join(self.weights_path, 'config_davis.json')
            cpu = False

        args = TempLoader()
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        self.cfg = load_config(args)

        self.siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, args.resume)
        self.siammask.eval().to(self.device)

    def track_bounding_box(self, item, annotation, frame_duration=60, progress=None):
        """

        :param item: dl.Item
        :param annotation: dl.Annotation
        :param frame_duration: How many frames ahead to track
        :param progress:
        :return:
        """
        try:
            assert isinstance(item, dl.Item)
            assert isinstance(annotation, dl.Annotation)
            item_stream_url = item.stream
            bbx = annotation.coordinates
            start_frame = annotation.start_frame

            logger.info('[Tracker] Started')
            tic_get_cap = time.time()
            logger.info('[Tracker] video url: {}'.format(item_stream_url))
            cap = cv2.VideoCapture('{}?jwt={}'.format(item_stream_url, dl.token()))
            runtime_get_cap = time.time() - tic_get_cap
            if not cap.isOpened():
                logger.error('[Tracker] failed opening video url')
                raise ValueError('cant open video stream. item id: {}'.format(item_stream_url))
            logger.info('[Tracker] received bbs(xyxy): {}'.format(bbx))
            tic_first_frame_set = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            runtime_first_frame_set = time.time() - tic_first_frame_set
            mask_enable = False
            runtime_load_frame = list()
            runtime_track = list()
            tic_total = time.time()
            state_bbx = None
            for i_frame in range(frame_duration):
                logger.info('[Tracker] processing frame #{}'.format(start_frame + i_frame))
                tic = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                runtime_load_frame.append(time.time() - tic)
                tic = time.time()
                # get bounding box
                top = bbx[0]['y']
                left = bbx[0]['x']
                bottom = bbx[1]['y']
                right = bbx[1]['x']
                w = right - left
                h = bottom - top
                if i_frame == 0:  # init
                    target_pos = np.array([left + w / 2, top + h / 2])
                    target_sz = np.array([w, h])
                    # init tracker
                    state_bbx = siamese_init(im=frame,
                                             target_pos=target_pos,
                                             target_sz=target_sz,
                                             model=self.siammask,
                                             hp=self.cfg['hp'],
                                             device=self.device)
                else:
                    # track
                    state_bbx = siamese_track(state=state_bbx,
                                              im=frame,
                                              mask_enable=mask_enable,
                                              refine_enable=True,
                                              device=self.device)
                    if mask_enable:
                        state_bbx['ploygon'].flatten()
                        mask = state_bbx['mask'] > state_bbx['p'].seg_thr

                        annotation.add_frame(
                            annotation_definition=dl.Box(top=np.min(np.where(mask)[0]),
                                                         left=np.min(np.where(mask)[1]),
                                                         bottom=np.max(np.where(mask)[0]),
                                                         right=np.max(np.where(mask)[1]),
                                                         label=annotation.label),
                            frame_num=start_frame + i_frame
                        )
                    else:
                        top = state_bbx['target_pos'][1] - state_bbx['target_sz'][1] / 2
                        left = state_bbx['target_pos'][0] - state_bbx['target_sz'][0] / 2
                        bottom = state_bbx['target_pos'][1] + state_bbx['target_sz'][1] / 2
                        right = state_bbx['target_pos'][0] + state_bbx['target_sz'][0] / 2

                        annotation.add_frame(
                            annotation_definition=dl.Box(top=top,
                                                         left=left,
                                                         bottom=bottom,
                                                         right=right,
                                                         label=annotation.label),
                            frame_num=start_frame + i_frame
                        )
                    runtime_track.append(time.time() - tic)
            runtime_total = time.time() - tic_total
            fps = frame_duration / runtime_total
            logger.info('[Tracker] Finished.')
            logger.info('[Tracker] Runtime information: \n'
                        'Total runtime: {:.2f}[s]\n'
                        'FPS: {:.2f}fps\n'
                        'Get url capture object: {:.2f}[s]\n'
                        'Initial set frame: {:.2f}[s]\n'
                        'Total track time: {:.2f}[s]\n'
                        'Mean load per frame: {:.2f}\n'
                        'Mean track per frame: {:.2f}'.format(runtime_total,
                                                              fps,
                                                              runtime_get_cap,
                                                              runtime_first_frame_set,
                                                              np.sum(runtime_load_frame) + np.sum(runtime_track),
                                                              np.mean(runtime_load_frame),
                                                              np.mean(runtime_track)))
            progress.update(status='success')
        except Exception:
            logger.exception('Failed during track:')
            raise
