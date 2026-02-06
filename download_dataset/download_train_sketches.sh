# 1️⃣ Create the final folder for the training sketches
mkdir -p /content/train_schetch

# 2️⃣ Path to the zip file
ZIP_PATH="/content/drive/MyDrive/datasets/shrec13/SBR/DataSets/SHREC13_SBR_TRAINING_SKETCHES.zip"

# 3️⃣ Extract only the necessary folders directly into /content/train_schetch_temp
unzip -o "$ZIP_PATH" -d /content/train_schetch_temp

# 4️⃣ Copy only .png files to /content/train_schetch while preserving class folders
for class_dir in /content/train_schetch_temp/SHREC13_SBR_TRAINING_SKETCHES/*; do
  if [ -d "$class_dir/train" ]; then
    class_name=$(basename "$class_dir")
    mkdir -p "/content/train_schetch/$class_name"
    cp "$class_dir/train/"*.png "/content/train_schetch/$class_name/"
  fi
done

# 5️⃣ Remove the temporary folder
rm -rf /content/train_schetch_temp