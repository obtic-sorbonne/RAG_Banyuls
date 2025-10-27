class DomainPrompts:
    @property
    def temperature_analysis(self):
        return (
            "Analyze the temperature data from the context. Identify: "
            "1. Daily min/max/mean temperatures "
            "2. Trends across the time period "
            "3. Anomalies compared to historical averages "
            "4. Potential correlations with other weather conditions mentioned"
        )
    
    @property
    def voyage_summary(self):
        return (
            "Create a comprehensive voyage summary including: "
            "1. Departure and arrival dates/locations "
            "2. Key weather events encountered "
            "3. Significant measurements taken "
            "4. Notable observations "
            "5. Crew signatures and positions"
        )
    
    @property
    def data_extraction(self):
        return (
            "Extract and structure the following information in JSON format: "
            "{"
            "  \"date\": \"YYYY-MM-DD\","
            "  \"water_temp\": \"value in °C\","
            "  \"air_temp\": \"value in °C\","
            "  \"salinity\": \"value in ‰\","
            "  \"wind_speed\": \"value in knots\","
            "  \"wind_direction\": \"compass direction\","
            "  \"observations\": \"key observations\""
            "}"
        )
