#!/bin/bash

# Author: Vojtěch Vlach
# Brief: Run PERO-OCR inference on specified folder of images
# Date: 10.10.2024
# Arguments:
#   $1: path to folder of images to infer
# Usage: ./run_pero_ocr.sh /path/to/images
# Description: Saves resulting data (page xmls and renders) in folders next to the images (for example /path/to/images -> /path/to/images/xml, /path/to/images/render)

# return $2 if $1 is empty (return default value if argument is not provided)
get_arg_or_default() {
    if ! [ -z "$1" ]; then
        echo $1
    else
        echo $2
    fi
}

# get abspath of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

IMAGES=$(get_arg_or_default $1 ${SCRIPT_DIR}/example_data/5_table_page_xmls_to_page_xmls/0_images/)
if ! [ -d "$IMAGES" ]; then
    echo "Directory $IMAGES does not exist. Provide a valid path as the first argument."
    exit 1
fi

XML_INPUT_PATH=$(get_arg_or_default $2 ${SCRIPT_DIR}/example_data/5_table_page_xmls_to_page_xmls/2_page_xmls)
if ! [ -d "$XML_INPUT_PATH" ]; then
    echo "Directory $XML_INPUT_PATH does not exist. Provide a valid path as the second argument."
    exit 1
fi

XML_OUT_PATH=$(get_arg_or_default $3 ${SCRIPT_DIR}/example_data/5_table_page_xmls_to_page_xmls/3_page_xmls_with_OCR)


RENDER_PATH=${XML_OUT_PATH}/render
# RENDER_LINE_PATH=${IMAGES}/render_line


files_in_image_dir=$(ls -p -1 $IMAGES | grep -v / | wc -l)
echo "Reading ${files_in_image_dir} image files from ${IMAGES}":
echo ""

# Run the OCR
python $PERO_OCR/user_scripts/parse_folder.py \
    -c $SCRIPT_DIR/pipeline_universal_ctc.ini \
    -i $IMAGES \
    --input-xml-path $XML_INPUT_PATH \
    --output-render-path $RENDER_PATH \
    --output-xml-path $XML_OUT_PATH \

    # --output-line-path $RENDER_LINE_PATH \

# print info about input and output files
files_exported=$(ls -p -1 $XML_OUT_PATH/*.xml | wc -l)

if [ -d "$XML_OUT_PATH" ]; then
    echo -e "${GREEN}Exported ${files_exported} files from ${files_in_image_dir} images${ENDCOLOR}"
else
    echo -e "${RED}No XML files were exported${ENDCOLOR}"
fi

