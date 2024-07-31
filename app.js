document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);

        fetch(form.action, {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: formData,
        })
        .then(response => response.text())  // Change this to text() to inspect the raw response
        .then(text => {
            console.log('Response Text:', text);
            let data;
            try {
                data = JSON.parse(text);  // Manually parse the text to JSON
            } catch (error) {
                console.error('Error parsing JSON:', error);
                document.getElementById('result').innerText = 'An error occurred. Please try again.';
                return;
            }
            console.log('Success:', data);
            if (data.result) {
                document.getElementById('result').innerText = `Predicted Obesity Level: ${data.result}`;
            } else {
                document.getElementById('result').innerText = 'Prediction failed. Please check your inputs.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerText = 'An error occurred. Please try again.';
        });
    });
});
