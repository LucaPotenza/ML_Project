# 1️⃣ Create the final destination folder
mkdir -p /content/evaluation

# 2️⃣ Path to the zip file
ZIP_PATH="/content/datasets/shrec13/SBR/SHREC2013_Sketch_Evaluation.zip"

# 3️⃣ Extract ONLY the SHREC13_SBR_Model.cla file into the final folder
unzip -j -o "$ZIP_PATH" "*SHREC13_SBR_Model.cla" -d /content/evaluation