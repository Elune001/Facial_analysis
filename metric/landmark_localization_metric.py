import numpy as np
from matplotlib import pyplot as plt


class landmark_Localization_Metric():
    '''
    landmark localization performance metric for batch

    배치 단위로 계산하는걸 기본으로 상정함

    만약 배치단위 아니면 이런식의 코드 필요함
    if np.ndim(pred_boxes) == 1 and len(pred_boxes) > 0:
        pred_boxes = pred_boxes[np.newaxis, :]

    usage
    nme = NME(landmark_num = 5)

    for data in generator:
        gt_bboxes = data['gt_bbox']
        gt_lands = data['gt_land']

        pred_bboxes = data['pred_bbox']
        pred_lands = data['pred_land']

        match_idx = matching_bbox(gt_bboxes,pred_bboxes)

        gt_bboxes = gt_bboxes[match_idx]
        pred_bboxes = pred_bboxes[match_idx]

        gt_lands = gt_lands[match_idx]
        pred_lands = pred_lands[match_idx]

        NME_norm_factor = nme.compute_NME_norm_factor(pred_bboxes)
        nme(gt_lands,pred_lands,NME_norm_factor)

    all_point_NME, five_point_NME= nme.get_result()

    '''

    def __init__(self,landmark_num = 5):
        self.NME_N_points_list = []
        self.NME_all_list = []
        self.landmark_num = landmark_num


    def compute_NME_norm_factor(self,bboxes):
        '''
        default norm factor

        :param bboxes: bounding box [batch,4]
        :return: NME norm factor
        '''
        bbox_hs = np.asarray(bboxes)[:, 3] - np.asarray(bboxes)[:, 1]
        bbox_ws = np.asarray(bboxes)[:, 2] - np.asarray(bboxes)[:, 0]
        NME_norm_factor = np.sqrt(bbox_hs * bbox_ws)
        return NME_norm_factor


    def NME_N_point(self,gt_lands,pred_lands,NME_norm_factor):
        # NME each key point [batch,5]
        NME_N_points = np.sqrt(
            np.sum(np.square(np.asarray(pred_lands) - np.asarray(gt_lands)),
                   axis=2)) / np.expand_dims(NME_norm_factor, axis=-1)
        return NME_N_points


    def NME_all_point(self,gt_lands,pred_lands,NME_norm_factor):
        # NME 5 key point [batch,]
        # NME norm factor : list type NME normalization factor
        NME_all = np.sum((np.sqrt(
            np.sum(np.square(np.asarray(pred_lands) - np.asarray(gt_lands)),
                   axis=2)) / np.expand_dims(NME_norm_factor, axis=-1)), axis=1) / self.landmark_num
        return NME_all

    def __call__(self,gt_lands,pred_lands,NME_norm_factor):
        # now NME is
        NME_all = self.NME_all_point(gt_lands,pred_lands,NME_norm_factor)
        NME_N_points = self.NME_N_point(gt_lands, pred_lands, NME_norm_factor)

        self.NME_N_points_list = self.NME_N_points_list + NME_N_points.tolist()
        self.NME_all_list = self.NME_all_list + NME_all.tolist()

        # return now NME
        return NME_all.tolist() ,NME_N_points.tolist()

    def get_result(self):
        N_point_NME = np.NaN
        if len(self.NME_N_points_list) != 0:
            N_point_NME = np.sum(np.asarray(self.NME_N_points_list), axis=0) / len(self.NME_N_points_list)
        all_point_NME = np.mean(np.asarray(self.NME_all_list))

        return all_point_NME,N_point_NME

    def get_landmark_failure(self,NME_threshold=0.1):
        # landmark_failure = np.sum(np.where(np.asarray(self.NME_all_list) > NME_threshold, 1, 0))
        # total_landmarks = len(self.NME_all_list)
        return self.landmark_failure_rate(self.NME_all_list,NME_threshold=NME_threshold)#landmark_failure/total_landmarks

    def get_AUC(self,NME_threshold=0.1):
        return self.AUC(self.NME_all_list, NME_threshold=NME_threshold, interval=0.001, show_CED=False)



    def landmark_failure_rate(self, NME_all_list, NME_threshold=0.1):
        landmark_failure = np.sum(np.where(np.asarray(NME_all_list) > NME_threshold, 1, 0))
        total_landmarks = len(NME_all_list)
        return landmark_failure / total_landmarks

    def AUC(self,NME_all_list, NME_threshold=0.1, interval=0.001, show_CED=False):
        '''
        :param NME_all_list: list type NME [NME1, NME2, NME3,...]
        :param NME_threshold: n of AUC_n
        :param interval: graph interval
        :return: AUC_n
        '''
        NMEs = NME_all_list
        total_face_num = len(NMEs)

        Y_axis = []
        Area = 0

        for threshold in np.arange(0.0, NME_threshold + interval, interval).tolist():
            Y_axis.append(np.sum(np.where(np.asarray(NMEs) < threshold, 1, 0)) / total_face_num)
            Area = Area + (np.sum(np.where(np.asarray(NMEs) < threshold, 1, 0)) / total_face_num) * interval

        AUC = Area / threshold

        # show up CED graph
        if show_CED == True:
            print('AUC: ', AUC)
            plt.plot(Y_axis)
            plt.title('CED')
            plt.ylabel('number of faces')
            plt.xlabel('NME')
            plt.legend(['Cumulative error curve'], loc='upper left')
            plt.show()

        return AUC






