from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm


def getIOU(polygon1: Polygon, polygon2: Polygon) -> float:
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    if union == 0:
        return 0
    return intersection / union


def compute_pq(gt_polygons: list, pred_polygons: list, iou_threshold=0.5):
    matched_instances = {}
    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))

    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))
    for gt_idx, gt_polygon in tqdm(enumerate(gt_polygons)):
        best_iou = iou_threshold
        best_pred_idx = None
        for pred_idx, pred_polygon in enumerate(pred_polygons):
            if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
                continue
            
            iou = getIOU(gt_polygon, pred_polygon)
            if iou == 0:
                continue
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_instances[(gt_idx, best_pred_idx)] = best_iou
            gt_matched[gt_idx] = 1
            pred_matched[best_pred_idx] = 1

    
    sq_sum = sum(matched_instances.values())
    num_matches = len(matched_instances)
    sq = sq_sum / num_matches if num_matches else 0
    rq = num_matches / (len(gt_polygons) + ((len(pred_polygons) - num_matches)/2.0)) if (gt_polygons or pred_polygons) else 0
    pq = sq * rq

    return pq, sq, rq



def test_compute_pq():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    pq, sq, rq = compute_pq(true_polygons, pred_polygons)
    assert pq == 0.6
    assert sq == 1
    assert rq == 0.6
    
def test_get_iou():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon2 = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
    
    assert getIOU(polygon1, polygon2) == 0.5
    
    
    