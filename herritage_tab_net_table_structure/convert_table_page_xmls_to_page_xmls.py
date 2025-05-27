#!/usr/bin/python3
"""Load table page XML files, create page layout and save it to XML files + render on a black canvas.
To not lose the table structure, it is saved in every textline as a custom JSON tag.

Usage:
$ python3 convert_table....py -x <xml_folder> -o <output_folder>
Resulting in:
    - <output_folder> with PAGE XML layouts with cell information in textline custom tag
    - <output_folder>/render with rendered images with textlines
"""

import argparse
import re
import sys
import os
import time
import logging
import cv2
from tqdm import tqdm
from bs4 import BeautifulSoup
from copy import deepcopy
from collections import defaultdict
import re
import json

import numpy as np

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding
from pero_ocr.core.layout import TextLine
from organizer.tables.table_layout import TableCell, TableRegion, TablePageLayout
from organizer.tables.rendering import render_cells, render_table_reconstruction
from organizer.utils import xywh_to_polygon


def parseargs():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-i", "--image-folder", default='example_data/2_cell_detection_tasks/images',
    #     help="Input folder where to look for images.")
    # parser.add_argument(
    #     "-l", "--label-file", type=str, default='example_data/4_annotated_HTML_tables.json',
    #     help="Label studio JSON export file.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/5_table_page_xmls_to_page_xmls/1_input_table_page_xmls',
        help="Folder with table page xml files with detected cells and table structure.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/5_table_page_xmls_to_page_xmls/2_page_xmls',
        help="Output folder where to save cut out objects.")
    # parser.add_argument(
    #     '-v', "--verbose", action='store_true', default=False,
    #     help="Activate verbose logging.")

    return parser.parse_args()


def main():
    args = parseargs()
    print(f'Running {os.path.basename(__file__)} with args: {args}\n{80*"-"}\n')
    start = time.time()

    os.makedirs(args.output_folder, exist_ok=True)
    render_dir = os.path.join(args.output_folder, 'render_on_black_canvas')
    os.makedirs(render_dir, exist_ok=True)

    xml_files = [xml_file for xml_file in os.listdir(args.xml_folder) if xml_file.endswith('.xml')]
    if len(xml_files) == 0:
        raise ValueError(f'No XML files found in {args.xml_folder}')
    
    xml_files_exported = 0

    for xml_file in tqdm(xml_files, unit='xml file'):
        xml_file_path = os.path.join(args.xml_folder, xml_file)
        if not os.path.exists(xml_file_path):
            raise ValueError(f'XML file {xml_file} does not exist in {args.xml_folder}')
        if not os.path.isfile(xml_file_path):
            raise ValueError(f'XML file {xml_file} is not a file in {args.xml_folder}')

        table_layout = TablePageLayout.from_table_pagexml(xml_file_path)

        page_layout = table_layout.to_page_layout_with_structure_in_custom_tag()

        # render layout to render_file_path
        canvas = np.zeros(page_layout.page_size + (3,), dtype=np.uint8)
        page_layout.render_to_image(canvas)
        render_file_path = os.path.join(render_dir, xml_file.replace('.xml', '.png'))
        cv2.imwrite(render_file_path, canvas)

        # save layout to XML file
        layout_file = os.path.join(args.output_folder, xml_file)
        page_layout.to_pagexml(layout_file)
        if not os.path.exists(layout_file):
            raise ValueError(f'Failed to save XML file {layout_file}')
        if not os.path.isfile(layout_file):
            raise ValueError(f'XML file {layout_file} is not a file in {args.output_folder}')
        xml_files_exported += 1


    print(f'Exported {xml_files_exported} XML files to {args.output_folder}')
    end = time.time()
    print(f'Total time: {end - start:.2f} s')

if __name__ == "__main__":
    main()

