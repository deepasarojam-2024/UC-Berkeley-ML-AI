# UC-Berkeley-ML-AI


Project: EDA - Will a Customer Accept the Coupon?
Overview

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those who did not.

Data

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then asks the person whether he will accept the coupon if he is the driver. Answers that the user will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’. There are five different types of coupons -- less expensive restaurants (under \$20), coffee houses, carry-out & takeaway bars, and more expensive restaurants (\$20 - \$50).

The attributes of this data set include:

**User attributes**
- Gender: male, female
- Age: below 21, 21 to 25, 26 to 30, etc.
- Marital Status: single, married partner, unmarried partner, or widowed
- Number of children: 0, 1, or more than 1
- Education: high school, bachelor's degree, associate degree, or graduate degree
- Occupation: architecture & engineering, business & financial, etc.
- Annual income: less than \$12500, \$12500 - \$24999, \$25000 - \$37499, etc.
- Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
- Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
- Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
- Number of times that he/she eats at a restaurant with average expense less than \$20 per person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
- Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8

**Contextual attributes**
- Driving destination: home, work, or no urgent destination
- Location of user, coupon, and destination: we provide a map to show the geographical location of the user, destination, and the venue, and we mark the distance between each two places with the time of driving. The user can see whether the venue is in the same direction as the destination.
- Weather: sunny, rainy, or snowy
- Temperature: 30F, 55F, or 80F
- Time: 10AM, 2PM, or 6PM
- Passenger: alone, partner, kid(s), or friend(s)

**Coupon attributes**
- time before it expires: 2 hours or one day

**Conclusion/Observations**
- Coffee-house coupons are offered the most followed by cheap restaurants (< $20).
- Most coupons were accepted at the highest temperature.
- Bar Coupons Acceptance has an acceptance rate of 41% and a non-acceptance rate of 59%. That means, there are slightly more chances of bar coupons getting not accepted in comparison to getting accepted.
- Bar coupons were accepted by those who went to the bar more than three times a month. (77% acceptance rate).
- Bar coupons were accepted by drivers with ages greater than 25 and who visited the bar more than once a month than others. (69.5% acceptance rate)
- Bar coupons were accepted by drivers who had no kids as passengers, didn't work in Farming Fishing, or forestry, and went to the bar more than once a month in comparison to others. (71% acceptance rate)
- Drivers who visited the bar more than once a month had no kids as passengers and were not windowed and had a 71% coupon acceptance rate.
- Drivers who visited the bar more than once a month, and have an age of less than 30 years, had a 64% coupon acceptance rate.
- Drivers who visited a cheap restaurant more than four times a month, and had income less than 50K, had a 41% coupon acceptance rate.
- People are accepting Cheap restaurants and carry-out coupons.
- Bar coupons are accepted, as we would expect.
- Bar Coupon acceptance is most common if people are with friends or partners.
- For expensive restaurants, people are not accepting coupons. People accept cheap restaurant coupons more.
- For Coffee House coupons, the acceptance and rejection rate is almost equal.
- Coupons should be sent in the evenings for more acceptance.
