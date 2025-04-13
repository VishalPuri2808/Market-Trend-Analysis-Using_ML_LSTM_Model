document.addEventListener("DOMContentLoaded", function() {
    const stockForm = document.getElementById("stockForm");
    const resultDiv = document.getElementById("result");
    const todayPredictionDiv = document.getElementById("todayPrediction");
    const sentimentDiv = document.getElementById("sentiment");
    const predictTrendBtn = document.getElementById("predictTrend");
    const stockInfoBtn = document.getElementById("stockInfo");

    let stockChartInstance = null;

    stockForm.addEventListener("submit", function(event) {
        event.preventDefault();
        predictStockPrice();
    });

    document.querySelectorAll(".tooltip button").forEach(button => {
        const tooltipText = button.getAttribute("data-tooltip");
        const tooltipSpan = button.nextElementSibling;
        tooltipSpan.innerText = tooltipText;
    });

    // document.getElementById("predictTrend").addEventListener("click", function() {
    //     const ticker = document.getElementById("ticker").value;

    //     fetch("/predict_trend", {
    //         method: "POST",
    //         headers: { "Content-Type": "application/x-www-form-urlencoded" },
    //         body: `ticker=${ticker}`
    //     })
    //     .then(response => response.json())
    //     .then(data => {
    //         trendPrediction.innerHTML = `<h3>Tomorrow's Trend: ${data.trend} (Predicted Price: $${data.tomorrow_price})</h3>`;
    //     });
    // });

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
            stockDetails.innerHTML = stockInfoHTML;
        });
    });

    function predictStockPrice() {
        const ticker = document.getElementById("ticker").value.trim();
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `ticker=${ticker}`
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `<h3 style='color:red;'>Error: ${data.error}</h3>`;
                return;
            }

            const predictions = data.predictions;
            const actualPrices = data.actual;
            const dates = data.dates;

            todayPredictionDiv.innerHTML = `<h3>Today's Predicted Price: $${data.predicted_price}</h3>`;
            sentimentDiv.innerHTML = `<h3>Market Sentiment: ${data.sentiment_score >= 0 ? 'Positive' : 'Negative'} (${data.sentiment_score})</h3>`;

            let newsHTML = "<h3>Recent News</h3><ul>";
            data.news.forEach(news => { newsHTML += `<li>${news}</li>`; });
            newsHTML += "</ul>";
            sentimentDiv.innerHTML += newsHTML;

            resultDiv.innerHTML = "<h3>Stock Price Prediction</h3><canvas id='stockChart'></canvas>";
            const ctx = document.getElementById('stockChart').getContext('2d');

            if (stockChartInstance) stockChartInstance.destroy();

            stockChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Actual Prices',
                            data: actualPrices.map(p => p[0]),
                            borderColor: 'blue',
                            fill: false,
                            pointRadius: 3,
                            pointBackgroundColor: 'blue',
                            borderWidth: 2
                        },
                        {
                            label: 'Predicted Prices',
                            data: predictions,
                            borderColor: 'green',
                            fill: false,
                            pointRadius: 3,
                            pointBackgroundColor: 'green',
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: { display: true, text: 'Date (MM/DD/YYYY)' },
                            ticks: { autoSkip: true, maxTicksLimit: 20 }
                        },
                        y: {
                            title: { display: true, text: 'Stock Price (USD)' },
                            beginAtZero: false
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error("Error fetching prediction:", error);
            resultDiv.innerHTML = `<h3 style='color:red;'>Error fetching prediction: ${error.message}</h3>`;
        });
    }
});
