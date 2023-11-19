import sys
from pathlib import Path
from typing import List

import numpy as np
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import list_files_recursively, load_json, save_json
from tqdm import tqdm


class labelme2coco:
    def __init__(self):
        raise RuntimeError(
            "Use labelme2coco.convert() or labelme2coco.get_coco_from_labelme_folder() instead."
        )


def get_coco_from_labelme_folder(
    labelme_folder: str,
    coco_category_list: List = None,
    skip_labels: List[str] = [],
) -> Coco:
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        coco_category_list: start from a predefined coco cateory list
    """
    # get json list
    _, abs_json_path_list = list_files_recursively(
        labelme_folder, contains=[".json"]
    )
    labelme_json_list = abs_json_path_list
    labelme_json_list.sort()

    # init coco object
    coco = Coco()
    coco.add_category(CocoCategory(id=1, name="xe"))
    coco.add_category(CocoCategory(id=2, name="ma"))
    coco.add_category(CocoCategory(id=3, name="tuong"))
    coco.add_category(CocoCategory(id=4, name="si"))
    coco.add_category(CocoCategory(id=5, name="vua"))
    coco.add_category(CocoCategory(id=6, name="phao"))
    coco.add_category(CocoCategory(id=7, name="tot"))

    if coco_category_list is not None:
        coco.add_categories_from_coco_category_list(coco_category_list)

    if len(skip_labels) > 0:
        print(f"Will skip the following annotated labels: {skip_labels}")

    # parse labelme annotations
    category_ind = 0
    for json_path in tqdm(
        labelme_json_list, "Converting labelme annotations to COCO format"
    ):
        data = load_json(json_path)
        # get image size
        image_path = str(Path(labelme_folder) / data["imagePath"])
        # use the image sizes provided by labelme (they already account for
        # things such as EXIF orientation)
        width = data["imageWidth"]
        height = data["imageHeight"]
        # init coco image
        coco_image = CocoImage(
            file_name=data["imagePath"], height=height, width=width
        )
        # iterate over annotations
        for shape in data["shapes"]:
            # set category name and id
            category_name = shape["label"]
            if category_name in skip_labels:
                continue
            category_id = None
            for (
                coco_category_id,
                coco_category_name,
            ) in coco.category_mapping.items():
                if category_name == coco_category_name:
                    category_id = coco_category_id
                    break

            # add category if not present
            # if category_id is None:
            #     category_id = category_ind
            #     coco.add_category(CocoCategory(id=category_id, name=category_name))
            #     category_ind += 1

            # skip if category_id is None
            if category_id is None:
                continue

            # circles and lines to segmentation
            if shape["shape_type"] == "circle":
                (cx, cy), (x1, y1) = shape["points"]
                r = np.linalg.norm(np.array([x1 - cx, y1 - cy]))
                angles = np.linspace(0, 2 * np.pi, 50 * (int(r) + 1))
                x = cx + r * np.cos(angles)
                y = cy + r * np.sin(angles)
                points = np.rint(np.append(x, y).reshape(-1, 2, order="F"))
                _, index = np.unique(points, return_index=True, axis=0)
                shape["points"] = points[np.sort(index)]
                shape["shape_type"] = "polygon"
            elif shape["shape_type"] == "line":
                (x1, y1), (x2, y2) = shape["points"]
                shape["points"] = [
                    x1,
                    y1,
                    x2,
                    y2,
                    x2 + 1e-3,
                    y2 + 1e-3,
                    x1 + 1e-3,
                    y1 + 1e-3,
                ]
                shape["shape_type"] = "polygon"

            # parse bbox/segmentation
            if shape["shape_type"] == "rectangle":
                x1 = shape["points"][0][0]
                y1 = shape["points"][0][1]
                x2 = shape["points"][1][0]
                y2 = shape["points"][1][1]
                coco_annotation = CocoAnnotation(
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    category_id=category_id,
                    category_name=category_name,
                )
            elif shape["shape_type"] == "polygon":
                segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                coco_annotation = CocoAnnotation(
                    segmentation=segmentation,
                    category_id=category_id,
                    category_name=category_name,
                )
            else:
                raise NotImplementedError(
                    f'shape_type={shape["shape_type"]} not supported.'
                )
            coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)

    return coco


if __name__ == "__main__":
    input_folder = sys.argv[1]  # test
    output_file = sys.argv[2]  # test.json
    coco = get_coco_from_labelme_folder(input_folder)
    save_json(coco.json, output_file)
