console.log('Loading file');

function load_script_async(uri) {
    return new Promise((resolve, reject) => {
        let script_element = document.createElement('script');
        script_element.src = uri;
        script_element.async = true;
        script_element.onload = () => {
            resolve();
        };
        let first_script = document.getElementsByTagName('script')[0];
        first_script.parentNode.insertBefore(script_element, first_script);
    });
}

function download_png(png, file_path) {
    let element = document.createElement('a');
    element.href = png;
    element.download = file_path;
    element.click();
}

function viewer_to_png(viewer_id, file_path) {
    download_png(viewer_dict[viewer_id].pngURI(), file_path);
}

function initialize_buffer(session_id) {
    buffer_dict[session_id] = [];
}

function push_to_buffer(session_id) {
    let buffer = buffer_dict[session_id];
    let viewer = viewer_dict[session_id];

    buffer.push(new Promise(resolve => {
        viewer.getCanvas().toBlob(function (blob) {
            blob.arrayBuffer().then(resolve);
        }, "image/png");
    }));
}

function process_buffer(session_id, delay, file_path) {
    let buffer = buffer_dict[session_id];
    let viewer = viewer_dict[session_id];

    let promise = new Promise(function (resolve, reject) {
        Promise.all(buffer).then(buffer => {
            let rga_list = [];

            // Pop items from buffer and convert them to RGBA
            while (buffer.length) {
                let item = buffer.pop();
                let img = UPNG.decode(item);
                rga_list.push(UPNG.toRGBA8(img)[0]);
            }

            if (rga_list.length === 0) {
                reject('Empty buffer');
            } else {
                let width = viewer.getCanvas().width;
                let height = viewer.getCanvas().height;
                let delays = Array(rga_list.length).fill(delay)

                // 0: all colors (lossless PNG)
                let apng = UPNG.encode(rga_list, width, height, 0, delays);
                let blob = new Blob([apng], {type: 'image/png'});
                let reader = new FileReader();
                reader.onload = function (e) {
                    resolve(e.target.result);
                };
                reader.readAsDataURL(blob);
            }
        });
    });

    promise.then(png => {
            download_png(png, file_path);
        }, // Success
        error => {
            console.log(error);
        }, // Failure
    );
}

function create_viewer(session_id, config) {
    let div_element = $('#' + session_id);
    if (!div_element) {
        throw 'Element with ID ' + session_id + 'not found';
    }
    let viewer = $3Dmol.createViewer(div_element, config);
    viewer_dict[session_id] = viewer;
    initialize_buffer(session_id);
    return viewer;
}

function delete_viewer(session_id) {
    delete viewer_dict[session_id];
    delete buffer_dict[session_id];
    $('#' + session_id).remove();
}

// Change element(s) to check if library loaded correctly
// Note: should not use id here as that is supposed to be a unique identifier
function change_element_of_class(class_name, inner) {
    const collection = document.getElementsByClassName(class_name);
    for (let i = 0; i < collection.length; i++) {
        collection[i].innerHTML = inner;
    }
}

// Load 3Dmol.js library
if (typeof $3Dmolpromise === 'undefined') {
    $3Dmolpromise = load_script_async("https://cdnjs.cloudflare.com/ajax/libs/3Dmol/1.8.0/3Dmol.min.js");
}

const buffer_dict = Object();
const viewer_dict = Object();
