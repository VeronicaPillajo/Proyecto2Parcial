$(function () {
    $('#add').on('click', function (e) {
        e.preventDefault();
        $('<div/>').addClass('new-text-div')
            .html($('<input type="textbox"/>').addClass('someclass'))
            .append($('<button/>').addClass('remove').text('Remove'))
            .insertBefore(this);
    });
    $(document).on('click', 'button.remove', function (e) {
        e.preventDefault();
        $(this).closest('div.new-text-div').remove();
    });
});
