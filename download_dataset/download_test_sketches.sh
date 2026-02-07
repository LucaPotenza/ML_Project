# 1️⃣ Create the final folder for the test sketches
mkdir -p /content/test_schetch

# 2️⃣ Path to the testing set zip file
ZIP_PATH="/content/datasets/shrec13/SBR/DataSets/SHREC13_SBR_TESTING_SKETCHES.zip"

# 3️⃣ Extract everything into a temporary folder
unzip -o "$ZIP_PATH" -d /content/test_schetch_temp

# 4️⃣ Copy only .png files to /content/test_schetch while preserving class folders
for class_dir in /content/test_schetch_temp/SHREC13_SBR_TESTING_SKETCHES/*; do
  if [ -d "$class_dir/test" ]; then
    class_name=$(basename "$class_dir")
    mkdir -p "/content/test_schetch/$class_name"
    cp "$class_dir/test/"*.png "/content/test_schetch/$class_name/"
  fi
done

# 5️⃣ Remove the temporary folder
rm -rf /content/test_schetch_temp