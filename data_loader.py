# These should be global, and populated once.
# I'll add checks in load_shrec13_data to ensure they are populated.
# label_names = []
# class_name_to_id = {}
# model_to_class = {}

# Global Loader

def load_shrec13_data(prm):
    """
    Carica sketch PNG per train e test e genera view da modelli 3D

    Returns:
        train_sketches: array (N_train, 10000) - immagini train 100×100 flatten
        test_sketches: array (N_test, 10000) - immagini test 100×100 flatten
        views: array (M, 10000) - view renderizzate 100×100 flatten
        train_sketch_labels: array (N_train, 1) - classe di ogni sketch train
        test_sketch_labels: array (N_test, 1) - classe di ogni sketch test
        view_labels: array (M, 1) - classe di ogni view
        label_names: list - nomi delle categorie (indice = class_id)
        class_name_to_id: dict - mappa nome_categoria → class_id
        model_to_class: dict - mappa model_id → class_id
    """
    global label_names, class_name_to_id, model_to_class

    # Initialize if not already initialized (e.g., if run in isolation)
    if 'label_names' not in globals():
        label_names = []
    if 'class_name_to_id' not in globals():
        class_name_to_id = {}
    if 'model_to_class' not in globals():
        model_to_class = {}

    # Path delle directory
    train_sketch_dir = Path('/content/train_schetch')
    test_sketch_dir = Path('/content/test_schetch')
    views_dir = Path('/content/views')

    # ===== BUILD LABEL NAMES MAPPING =====
    class_file = Path('/content/evaluation/SHREC13_SBR_Model.cla')


    if class_file.exists():
        with open(class_file, 'r') as f:
            lines = f.readlines()

        # Salta le prime 2 righe (header)
        i = 2
        current_class_id = -1
        num_models_remaining = 0

        while i < len(lines):
            line = lines[i].strip()

            # Salta righe vuote o solo spazi
            if not line:
                i += 1
                continue

            parts = line.split()

            # Se la riga ha 3 parti: category_name 0 num_models
            if len(parts) == 3 and parts[1] == '0':
                category_name = parts[0]  # es: "airplane"
                num_models = int(parts[2])  # es: 184

                # Assegna un class_id progressivo
                current_class_id = len(label_names)
                label_names.append(category_name)
                class_name_to_id[category_name.lower()] = current_class_id

                num_models_remaining = num_models

            # Altrimenti è un model_id
            elif len(parts) == 1 and parts[0].isdigit():
                model_id = parts[0]
                if current_class_id >= 0 and num_models_remaining > 0:
                    model_to_class[model_id] = current_class_id
                    num_models_remaining -= 1

            i += 1
    else:
        raise FileNotFoundError(f"Class file not found: {class_file}")

    # ===== CARICA SKETCH TRAIN E TEST =====
    train_sketches, train_sketch_labels = load_sketches_from_dir(train_sketch_dir, prm)

    test_sketches, test_sketch_labels = load_sketches_from_dir(test_sketch_dir, prm)

    views_list, view_labels_list = load_views_from_dir(views_dir, prm)

    views = np.array(views_list)
    view_labels = np.array(view_labels_list).reshape(-1, 1)

    return train_sketches, test_sketches, views, train_sketch_labels, test_sketch_labels, view_labels, label_names, class_name_to_id, model_to_class

# FUNZIONE PER CARICARE VIEWS DA UNA DIRECTORY
def load_views_from_dir(views_dir, prm):
    views_list = []
    view_labels_list = [] # Initialize view_labels_list

    # Carica tutti i PNG nella cartella
    png_files = sorted(views_dir.glob('*.png'))

    for png_file in png_files:
        # Leggi immagine in grayscale
        img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        shape = img.shape

        if shape[0] != prm.inputWH or shape[1] != prm.inputWH:
          img = cv2.resize(img, (prm.inputWH, prm.inputWH))
        # controllare meglio il resizing se non è troppo

        # Normalizza [0, 255] → [0, 1]
        img = img.astype(np.float32) / 255.0

        # Flatten
        img_flat = img.flatten()

        views_list.append(img_flat)

        # Corrected line: Use png_file.name to get the string representation
        model_id = png_file.name.split('_')[0]

        # Remove initial "m" if present
        if model_id.startswith('m'):
            model_id = model_id[1:]

        view_labels_list.append(model_to_class[model_id])

    views = np.array(views_list)
    views_labels = np.array(view_labels_list).reshape(-1, 1)

    return views, views_labels

# FUNZIONE PER CARICARE SKETCH DA UNA DIRECTORY
def load_sketches_from_dir(sketch_dir, prm):
    sketches_list = []
    sketch_labels_list = []

    # Itera sulle cartelle di classe
    for class_folder in sorted(sketch_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name.lower()  # es: "airplane"

        # Ottieni class_id dalla mappa
        class_id = class_name_to_id.get(class_name, -1)

        if class_id == -1:
            print(f"Warning: categoria '{class_name}' non trovata in .cla file")
            continue

        # Carica tutti i PNG nella cartella
        png_files = sorted(class_folder.glob('*.png'))

        for png_file in png_files:
            # Leggi immagine in grayscale
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Resize a 100×100
            img = cv2.resize(img, (prm.inputWH, prm.inputWH))
 # controllare meglio il resizing se non è troppo

            # Normalizza [0, 255] → [0, 1]
            img = img.astype(np.float32) / 255.0

            # Flatten
            img_flat = img.flatten()

            sketches_list.append(img_flat)
            sketch_labels_list.append(class_id)

    sketches = np.array(sketches_list)
    sketch_labels = np.array(sketch_labels_list).reshape(-1, 1)

    return sketches, sketch_labels