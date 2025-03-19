document.addEventListener("DOMContentLoaded", function() {
    const stockForm = document.getElementById("stockForm");
    const resultDiv = document.getElementById("result");
    const todayPredictionDiv = document.getElementById("todayPrediction");
    const trendPredictionDiv = document.getElementById("trendPrediction");
    const stockDetailsDiv = document.getElementById("stockDetails");

    // Tooltip Functionality
    document.querySelectorAll(".tooltip button").forEach(button => {
        const tooltipText = button.getAttribute("data-tooltip");
        const tooltipSpan = button.nextElementSibling;
        tooltipSpan.innerText = tooltipText;
    });

    // Predict Stock Prices
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
                            borderColor: 'blue'
                        },
                        {
                            label: 'Predicted Prices',
                            data: predictions.map(p => p[0]),
                            borderColor: 'green'
                        }
                    ]
                }
            });
        });
    });

    // Predict Tomorrow's Trend
    document.getElementById("predictTrend").addEventListener("click", function() {
        const ticker = document.getElementById("ticker").value;

        fetch("/predict_trend", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `ticker=${ticker}`
        })
        .then(response => response.json())
        .then(data => {
            trendPredictionDiv.innerHTML = `<h3>Tomorrow's Trend: ${data.trend} (Predicted Price: $${data.tomorrow_price})</h3>`;
        });
    });

    // Fetch Stock Information
    document.getElementById("stockInfo").addEventListener("click", function() {
        const ticker = document.getElementById("ticker").value;

        fetch("/stock_info", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `ticker=${ticker}`
        })
        .then(response => response.json())
        .then(data => {
            let stockInfoHTML = `
                <h3>Stock Information</h3>
                <table class="stock-table">
                    <tr><th>Company Name</th><td>${data["Company Name"]}</td></tr>
                    <tr><th>Market Cap</th><td>${data["Market Cap"].toLocaleString()}</td></tr>
                    <tr><th>PE Ratio</th><td>${data["PE Ratio"]}</td></tr>
                    <tr><th>52 Week High</th><td>${data["52 Week High"]}</td></tr>
                    <tr><th>52 Week Low</th><td>${data["52 Week Low"]}</td></tr>
                </table>
            `;

            stockDetailsDiv.innerHTML = stockInfoHTML;
        });
    });
});
