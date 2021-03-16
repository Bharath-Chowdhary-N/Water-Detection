function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            var image = document.createElement("img");
            //var image=new Image(300,300);
            image.setAttribute("height", "300");
            image.setAttribute("width", "300");
            image.setAttribute("alt", "input_image");
             


            // the result image data
            image.src = e.target.result;
            var src = document.getElementById("header");
            src.appendChild(image)
            //document.getElementById("img").appendChild(image);
            //document.getElementById("demo").innerHTML = image;
            //document.body.appendChild(image);
            $('#blah')
                .attr('src', e.target.result)
                .width(300)
                .height(300);
            $('#blah').click(function() {
                    location.reload();
            });
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function readIMG(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            var output = document.getElementById('blah')
            output.src = reader.result;
            $('#blah').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
    }
}

//$("#imgInp").change(function(){
//    readIMG(this);
//});