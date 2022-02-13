
## ------- CONFIG -------
# Load and install necessary packages
requiredPackages <- c("readr",
                      "lubridate")

for (pac in requiredPackages) {
    if(!require(pac,  character.only=TRUE)){
        install.packages(pac, repos="http://cran.rstudio.com")
        library(pac,  character.only=TRUE)
    } 
}
rm(pac)
rm(requiredPackages)

## ------- WORKING PATH -------
# Our working directories.
wd = getwd()
if(grepl("nora", wd)) {
    setwd("/home/nora/git/adm-project/report01")
    Sys.setlocale("LC_TIME", "en_US.utf8")   # Modern Linux etc.
} else {
    setwd("~/Desktop/ADM/adm-project/report01")
    Sys.setlocale("LC_TIME", "en_US")  # OS X, in UTF-8
}
rm(wd)



## ------- FUNCTIONS -------
getIndexByName <- function(df, name) {
    return (grep(name, colnames(df)))
}

removeColsByName <- function(df, columnNames) {
    indexes = c()
    for(col in columnNames) {
        indexes = append(indexes, getIndexByName(df, col))
    } 
    return (subset(df, select = -indexes))
}


## ------- PREPROCESSING -------
df <- read_delim("./newDatasets/final_merged_df.csv", ";", escape_double = FALSE, trim_ws = TRUE)
#View(df)

# Removing unused columns
df = removeColsByName(df, c("X1", "Accident_Index", "Longitude", "Latitude" , "worstCasualtySeverity"))

final_df = df

# Use only the hour, removing the minutes...
hours =  hour(as.POSIXct(strptime(final_df$Time, "%H:%M:%S")))
head(hours)
final_df$Time = hours


# Keep month and dey of the week

date = as.Date(final_df$Date, "%d/%m/%Y")
months = format(date, "%m")

dayOfWeek =  wday(date, label=TRUE)
dayOfWeek = as.character(dayOfWeek)

isWeekend = dayOfWeek
length(dayOfWeek)
for (i in seq(1, length(dayOfWeek))) {
    isWeekend[i] = ifelse(dayOfWeek[i] == "Sun" || dayOfWeek[i] == "Sat", "WEEKEND", "WORKDAY")
}

final_df = cbind(final_df, Is_Weekend=isWeekend)
final_df = cbind(final_df, Day_of_Week = dayOfWeek)
final_df = cbind(final_df, Month = months)

View(final_df)


## Categorical values instead of numerical ones
accident_severity = c("FATAL", "SERIOUS", "SLIGHT")
road_type = c("Roundabout", "One_Way_Street", "Dual_Carriageway", "", "", "Single_Carriageway", "Slip_Road", "", "Unknown", "", "", "One_way/Slip_Road")
junction_detail = c("No_junction", "Roundabout", "Mini_Roundabout", "T", "Slip_Road", "Crossroads", "More_4_Arms", "Private_entrance", "Other_junction")
vehicle_type = c("Motorcycle", "car", "trucks", "animals", "mixed")
worst_casualty_severity = c("Fatal", "Serious", "Slight")
police_attended = c("Yes", "No", "No-SelfCompletionForm")


final_df$Accident_Severity = accident_severity[final_df$Accident_Severity]
final_df$Road_Type = road_type[final_df$Road_Type]
final_df$Junction_Detail = junction_detail[final_df$Junction_Detail+1]
final_df$involvedVehType = vehicle_type[final_df$involvedVehType]
# final_df$worstCasualtySeverity = worst_casualty_severity[final_df$worstCasualtySeverity]
final_df$Did_Police_Officer_Attend_Scene_of_Accident = police_attended[final_df$Did_Police_Officer_Attend_Scene_of_Accident]


# CATEGORICAL VALUES for each case
final_df$Light_Conditions = ifelse(final_df$Light_Conditions == "GOOD", "GOOD_LIGHT", "BAD_LIGHT")
final_df$Weather_Conditions = ifelse(final_df$Weather_Conditions == "GOOD", "GOOD_WEATHER", "BAD_WEATHER")
final_df$Road_Surface_Conditions = ifelse(final_df$Road_Surface_Conditions == "GOOD", "GOOD_ROAD", "BAD_ROAD")
final_df$Special_Conditions_at_Site = ifelse(final_df$Special_Conditions_at_Site == "TRUE", "SPECIAL_COND", "NO_SPECIAL_COND")
final_df$Carriageway_Hazards = ifelse(final_df$Carriageway_Hazards == "TRUE", "ROAD_HAZARD", "NO_ROAD_HAZARD")
final_df$Did_Police_Officer_Attend_Scene_of_Accident = ifelse(final_df$Did_Police_Officer_Attend_Scene_of_Accident == "Yes", "POLICE_ATTENDED", "NO_POLICE")

# Remove date column
final_df = removeColsByName(final_df, c("Date"))

# Remove NA.
nrow(final_df)
cleanDF = final_df[complete.cases(final_df), ]
nrow(cleanDF)
#final_df[is.na(final_df)] <- " "

View(cleanDF)
write.csv2(cleanDF, file = "../report02/naive_bayes_dataset.csv", row.names=FALSE)

