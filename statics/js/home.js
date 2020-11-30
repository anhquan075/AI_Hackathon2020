// var subdir_fold = None;

$('#btn-savefile').click(function () {
    var id_file = document.getElementById("id_file").value;
    // var id_fold = document.getElementById("id_fold").value;
    // var filename = document.getElementById("filename").value;

    saveZip(id_file);
}
)

$('#btn-runsystem').click(function () {
    var filename = document.getElementById("filename").value;
    var id_fold = document.getElementById("id_fold").value;

    unZip(filename, id_fold);
}
)

function saveZip(id_file){
    document.getElementById("noti_save").innerHTML = 'Downloading!';
    $.ajax({
        url: 'save_zip?id_file='+id_file,
        type: 'get',
        dataType: 'json',
        contentType: 'application/json',  
        success: function (response) {
            if (response['code'] == 1001) {
                alert("[Lỗi] Không nhận được phản hồi từ server, vui lòng kiểm tra lại!");
            }
            console.log(response);

            document.getElementById("noti_save").innerHTML = 'Download done!';
        }
    }).done(function() {
        
    }).fail(function() {
        alert('Fail!');
    });
}

function unZip(filename, id_fold){
    document.getElementById("noti_run").innerHTML = 'Now unzip!';
    $.ajax({
        url: 'unzip?filename='+filename+'&id_fold='+id_fold,
        type: 'get',
        dataType: 'json',
        contentType: 'application/json',  
        success: function (response) {
            if (response['code'] == 1001) {
                alert("[Lỗi] Không nhận được phản hồi từ server, vui lòng kiểm tra lại!");
            }
            console.log(response);
            subdir = response['subdir']
            document.getElementById("noti_run").innerHTML = 'Predicting......!';
            predict(subdir)
        }
    }).done(function() {
        
    }).fail(function() {
        alert('Fail!');
    });
}

function predict(subdir){
    document.getElementById("noti_predict").innerHTML = 'Predicting......!';
    $.ajax({
        url: 'predict?subdir='+subdir,
        type: 'get',
        dataType: 'json',
        contentType: 'application/json',  
        success: function (response) {
            if (response['code'] == 1001) {
                alert("[Lỗi] Không nhận được phản hồi từ server, vui lòng kiểm tra lại!");
            }
            console.log(response);
            out_sub_path = response['out_sub_path']
            out_sub_path_vid = response['out_sub_path_vid']

            document.getElementById("out_sub_path").innerHTML = out_sub_path;
            document.getElementById("out_sub_path_vid").innerHTML = out_sub_path_vid;
            document.getElementById("my_time").innerHTML = str(my_time);
        }
    }).done(function() {
        
    }).fail(function() {
        alert('Fail!');
    });
}