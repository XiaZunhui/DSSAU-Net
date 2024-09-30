import numpy
import numpy as np
import cv2
import math
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef


class Biometry:
    def __init__(self):
        pass

    def onehot_to_mask(self, mask):
        ret = np.zeros([3, 512, 512])
        tmp = mask.copy()
        tmp[tmp == 1] = 255
        tmp[tmp == 2] = 0
        ret[1] = tmp
        tmp = mask.copy()
        tmp[tmp == 2] = 255
        tmp[tmp == 1] = 0
        ret[2] = tmp
        b = ret[0]
        r = ret[1]
        g = ret[2]
        ret = cv2.merge([b, r, g])
        mask = ret.transpose([0, 1, 2])
        return mask

    def get_ps_fh_points(self, image, part="all"):
        """
        getting contours
        :param path:
        :param part:
        :return:
        """
        # pred = sitk.ReadImage(path)
        # pred_data = sitk.GetArrayFromImage(pred)  # 0-8-7 three values

        aop_pred = np.array(self.onehot_to_mask(image)).astype(np.uint8)  # 0 background; 1 ps; 2 fh
        contours, _ = cv2.findContours(aop_pred[:, :, 1], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(aop_pred[:, :, 2], cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        max_contour_p_num = 0
        max_contour2_p_num = 0
        for i in range(len(contours)):
            p_num = contours[i].shape[0]
            if p_num > max_contour_p_num:
                max_contour_p_num = p_num
                contour_dim_index = i
        for i in range(len(contours2)):
            p_num = contours2[i].shape[0]
            if p_num > max_contour2_p_num:
                max_contour2_p_num = p_num
                contour2_dim_index = i

        points_ps_transpose = [tuple(contours[contour_dim_index][i][0]) for i in
                               range(len(contours[contour_dim_index]))]  # (x,y) pair  support for opencv
        points_fh_transpose = [tuple(contours2[contour2_dim_index][i][0]) for i in
                               range(len(contours2[contour2_dim_index]))]  # (x,y) pair

        points_ps = [(i[1], i[0]) for i in points_ps_transpose]  # (y,x) pair  support for numpy array
        points_fh = [(i[1], i[0]) for i in points_fh_transpose]
        # print(points_ps)
        if part == "ps":
            ret = {"ps": points_ps}
        if part == "fh":
            ret = {"fh": points_fh}
        if part == "all":
            ret = {"ps": points_ps, "fh": points_fh}
        return ret

    def get_points_on_ps(self, image):
        """
        getting two points on ps
        :param path:
        :return:
        """
        ps = self.get_ps_fh_points(image, part="ps")['ps']  # got (x,y) pair
        max_len = 0
        max_pair = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d = (ps[i][0] - ps[j][0]) ** 2 + (ps[i][1] - ps[j][1]) ** 2
                if d > max_len:
                    max_len = d
                    max_pair = [ps[i], ps[j]]
        return max_pair

    def get_hsd_point_on_fh(self, image):
        """
        getting hsd point on fh
        :param path:
        :return:
        """
        ps = self.get_points_on_ps(image)
        try:
            ps_point = ps[0] if ps[0][1] > ps[1][1] else ps[1]  # (y,x) pair, choose max x
        except:
            pass

        fh = self.get_ps_fh_points(image, part='fh')['fh']
        hsd_point = []
        for i in range(len(fh)):
            d = (fh[i][0] - ps_point[0]) ** 2 + (fh[i][1] - ps_point[1]) ** 2
            if i == 0:
                min_len = d
                hsd_point = fh[i]
            elif d < min_len:
                min_len = d
                hsd_point = fh[i]
        return hsd_point

    def position(self, ps_point, p, fh):
        fh_sort = sorted(fh, key=lambda x: int(x[1]), reverse=True)
        flag = True
        addition_tangency = []
        if ps_point[1] != p[1]:
            K = 1.0 * (ps_point[0] - p[0]) / (ps_point[1] - p[1])
            # print(K)
            for p_fh in fh_sort:
                if p_fh != p:
                    s = K * (p_fh[1] - ps_point[1]) + ps_point[0] - p_fh[0]  # >0 right  <0left
                    if (s > 0 and K >= 0) or (s < 0 and K < 0):
                        flag = False
                        break
                    if s == 0:
                        addition_tangency.append(p_fh)
                else:
                    continue
        else:
            for p_fh in fh:
                if p_fh != p:
                    if p_fh[1] > p[1]:
                        flag = False
                        break
                    elif p_fh[1] == p[1]:
                        addition_tangency.append(p_fh)
        return flag, addition_tangency

    def get_tangency(self, image):
        ps = self.get_points_on_ps(image)
        ps_point = ps[0] if ps[0][1] > ps[1][1] else ps[1]
        fh = self.get_ps_fh_points(image, part='fh')['fh']
        tangency = []
        fh = sorted(fh, key=lambda x: int(x[1]), reverse=True)
        for p in fh:
            flag, add_tangency = self.position(ps_point, p, fh)
            if flag:
                if tangency == []:
                    min_d = (ps_point[0] - p[0]) ** 2 + (ps_point[1] - p[1]) ** 2
                    tangency = p
                if add_tangency != []:
                    for add_p in add_tangency:
                        d = (ps_point[0] - add_p[0]) ** 2 + (ps_point[1] - add_p[1]) ** 2
                        if d < min_d:
                            min_d = d
                            tangency = add_p
        return tangency

    def extract_landmarks(self, image):
        """
        :param image: numpy array (H,W) with value 0-1-2, 1 represent PS 2, represent FH
        :return:
        """

        ps_points = self.get_points_on_ps(image)
        hsd_point = self.get_hsd_point_on_fh(image)
        tangency = self.get_tangency(image)
        if tangency == []:
            print(f"no tangency found")
        return ps_points, hsd_point, tangency

    def calc_aop_hsd(self, image):
        ps_points, hsd_point, tangency = self.extract_landmarks(image)

        ps_point = ps_points[0] if ps_points[0][1] > ps_points[1][1] else ps_points[1]
        ps_point2 = ps_points[1] if ps_points[0][1] > ps_points[1][1] else ps_points[0]
        hsd = math.sqrt((ps_point[0] - hsd_point[0]) ** 2 + (ps_point[1] - hsd_point[1]) ** 2)

        a = math.sqrt((ps_point[0] - ps_point2[0]) ** 2 + (ps_point[1] - ps_point2[1]) ** 2)
        b = math.sqrt((ps_point[0] - tangency[0]) ** 2 + (ps_point[1] - tangency[1]) ** 2)
        c = math.sqrt((ps_point2[0] - tangency[0]) ** 2 + (ps_point2[1] - tangency[1]) ** 2)
        value = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

        aop = math.acos(value) * 180 / math.pi

        res = {
            "ps_points": ps_points,
            "hsd_point": hsd_point,
            "aop_tangency": tangency,
            "hsd": hsd,
            "aop": aop,
        }
        return res

    def process(self, pred, label):
        """

        :param pred: numpy array (H,W) with value 0-1-2, 1 represent PS 2, represent FH
        :param label: numpy array (H,W) with value 0-1-2, 1 represent PS 2, represent FH
        :return:
        """
        res1 = self.calc_aop_hsd(pred)
        res2 = self.calc_aop_hsd(label)
        res = {
            "delta": {"aop": abs(res1['aop'] - res2['aop']), "hsd": abs(res1['hsd'] - res2['hsd'])},
            "pred": res1,
            "gt": res2,
        }
        return res


def demo_biometry():
    img = sitk.ReadImage("../dataset_sample/pos/20190909T155747I1/mask/20190909T155747I1_0_6.png")
    img = sitk.GetArrayFromImage(img)
    img[np.where(img == 8)] = 2
    img[np.where(img == 7)] = 1

    eva = Biometry()
    res = eva.process(img)
    print(res)


class Segment:
    def __init__(self):
        pass

    def numpy_to_image(self, image) -> sitk.Image:
        image = sitk.GetImageFromArray(image)
        return image

    def evaluation(self, pred: sitk.Image, label: sitk.Image):
        result = dict()
        # 计算耻骨指标
        pred_data_ps = sitk.GetArrayFromImage(pred)
        pred_data_ps[pred_data_ps == 2] = 0
        pred_ps = sitk.GetImageFromArray(pred_data_ps)

        label_data_ps = sitk.GetArrayFromImage(label)
        label_data_ps[label_data_ps == 2] = 0
        label_ps = sitk.GetImageFromArray(label_data_ps)
        if (pred_data_ps == 0).all():
            result['asd_ps'] = 100.0
            result['dice_ps'] = 0.0
            result['hd_ps'] = 100.0
        else:
            result['asd_ps'] = float(self.cal_asd(pred_ps, label_ps))
            result['dice_ps'] = float(self.cal_dsc(pred_ps, label_ps))
            result['hd_ps'] = float(self.cal_hd(pred_ps, label_ps))

        # 计算胎头指标
        pred_data_head = sitk.GetArrayFromImage(pred)
        pred_data_head[pred_data_head == 1] = 0
        pred_data_head[pred_data_head == 2] = 1
        pred_head = sitk.GetImageFromArray(pred_data_head)

        label_data_head = sitk.GetArrayFromImage(label)
        label_data_head[label_data_head == 1] = 0
        label_data_head[label_data_head == 2] = 1
        label_head = sitk.GetImageFromArray(label_data_head)

        if (pred_data_head == 0).all():
            result['asd_fh'] = 100.0
            result['dice_fh'] = 0.0
            result['hd_fh'] = 100.0
        else:
            result['asd_fh'] = float(self.cal_asd(pred_head, label_head))
            result['dice_fh'] = float(self.cal_dsc(pred_head, label_head))
            result['hd_fh'] = float(self.cal_hd(pred_head, label_head))

        # 计算总体指标
        pred_data_all = sitk.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = sitk.GetImageFromArray(pred_data_all)

        label_data_all = sitk.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = sitk.GetImageFromArray(label_data_all)
        if (pred_data_all == 0).all():
            result['asd_all'] = 100.0
            result['dice_all'] = 0.0
            result['hd_all'] = 100.0
        else:
            result['asd_all'] = float(self.cal_asd(pred_all, label_all))
            result['dice_all'] = float(self.cal_dsc(pred_all, label_all))
            result['hd_all'] = float(self.cal_hd(pred_all, label_all))
        return result

    def process(self, pred: numpy.array, label: numpy.array):
        """

        :param pred: (H,W) value:0,1,2  1-ps 2-fh
        :param label: (H,W) value:0,1,2  1-ps 2-fh
        :return:
        """
        pre_image = self.numpy_to_image(pred)
        truth_image = self.numpy_to_image(label)
        result = self.evaluation(pre_image, truth_image)

        return result

    def cal_asd(self, a, b):
        filter1 = sitk.SignedMaurerDistanceMapImageFilter()  # 于计算二值图像中像素到最近非零像素距离的算法
        filter1.SetUseImageSpacing(True)  # 计算像素距离时要考虑像素之间的间距
        filter1.SetSquaredDistance(False)  # 计算距离时不要对距离进行平方处理
        a_dist = filter1.Execute(a)
        a_dist = sitk.GetArrayFromImage(a_dist)
        a_dist = np.abs(a_dist)
        a_edge = np.zeros(a_dist.shape, a_dist.dtype)
        a_edge[a_dist == 0] = 1
        a_num = np.sum(a_edge)

        filter2 = sitk.SignedMaurerDistanceMapImageFilter()
        filter2.SetUseImageSpacing(True)
        filter2.SetSquaredDistance(False)
        b_dist = filter2.Execute(b)

        b_dist = sitk.GetArrayFromImage(b_dist)
        b_dist = np.abs(b_dist)
        b_edge = np.zeros(b_dist.shape, b_dist.dtype)
        b_edge[b_dist == 0] = 1
        b_num = np.sum(b_edge)

        a_dist[b_edge == 0] = 0.0
        b_dist[a_edge == 0] = 0.0

        asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

        return asd

    def cal_dsc(self, pd, gt):
        pd = sitk.GetArrayFromImage(pd).astype(np.uint8)
        gt = sitk.GetArrayFromImage(gt).astype(np.uint8)
        y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
        return y

    def cal_hd(self, a, b):
        a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
        b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
        filter1 = sitk.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd


def demo_segment():
    pred = "../dataset_sample/pos/20190909T155747I1/mask/20190909T155747I1_0_6.png"
    gt = "../dataset_sample/pos/20190909T155747I1/mask/20190909T155747I1_9_6.png"
    pred = sitk.ReadImage(pred)
    pred = sitk.GetArrayFromImage(pred)
    pred[np.where(pred == 8)] = 2
    pred[np.where(pred == 7)] = 1
    gt = sitk.ReadImage(gt)
    gt = sitk.GetArrayFromImage(gt)
    gt[np.where(gt == 8)] = 2
    gt[np.where(gt == 7)] = 1

    seg = Segment()
    res = seg.process(pred, gt)
    print(res)


class Classification:
    def __init__(self):
        pass

    def process(self, pred, gt):
        """

        :param pred: (B,C)
        :param gt:  (B,)
        :return:
        """

    def accuracy(self, pred, gt):
        pred_t = pred.argmax(-1)
        total = pred_t.shape[0]
        correct = (pred_t == gt).sum()
        return correct / total

    def auc(self, pred, gt):
        pred_t = pred[:, 1]
        auc_value = roc_auc_score(gt, pred_t)
        return auc_value

    def f1_score(self, pred, gt):
        pred_t = pred.argmax(-1)
        f1_value = f1_score(gt, pred_t, average='binary')
        return f1_value

    def matthews_correlation_coef(self, pred, gt):
        gt[np.where(gt == 0)] = -1
        pred_t = pred.argmax(-1)
        pred_t[np.where(pred_t == 0)] = -1
        mcc = matthews_corrcoef(gt, pred_t)
        return mcc


if __name__ == '__main__':
    c = Classification()
    pred = np.array([
        [0.3, 0.7], [0.8, 0.2], [0.01, 0.99], [1, 0]
    ])
    gt = np.array([0, 1, 1, 0])
    res = c.auc(pred, gt)
    print(res)
