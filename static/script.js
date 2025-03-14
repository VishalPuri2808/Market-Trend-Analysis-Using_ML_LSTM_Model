document.addEventListener("DOMContentLoaded", function() {
    const stockForm = document.getElementById("stockForm");
    const resultDiv = document.getElementById("result");
    const todayPredictionDiv = document.getElementById("todayPrediction");

    stockForm.addEventListener("submit", function(event) {
        event.preventDefault();
        const ticker = document.getElementById("ticker").value;

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `ticker=${ticker}`
        })
        .then(response => response.json())
        .then(data => {
            const predictions = data.predictions;
            const actualPrices = data.actual;
            const dates = data.dates;
            const todayPrediction = predictions[predictions.length - 1][0];
            const todayDate = dates[dates.length - 1];

            todayPredictionDiv.innerHTML = `<h3>Prediction for Today (${todayDate}): $${todayPrediction.toFixed(2)}</h3>`;

            resultDiv.innerHTML = "<h3>Stock Price Prediction</h3><canvas id='stockChart'></canvas>";

            const ctx = document.getElementById('stockChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Actual Prices',
                            data: actualPrices.map(p => p[0]),
                            borderColor: 'blue',
                            fill: false,
                            pointStyle: 'circle',
                            pointRadius: 3,
                            pointBackgroundColor: 'blue'
                        },
                        {
                            label: 'Predicted Prices',
                            data: predictions.map(p => p[0]),
                            borderColor: 'green',
                            fill: false,
                            pointStyle: 'circle',
                            pointRadius: 3,
                            pointBackgroundColor: 'green'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date (MM/DD/YYYY)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Stock Price (USD)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                color: 'black'
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error("Error fetching prediction:", error));
    });
});
