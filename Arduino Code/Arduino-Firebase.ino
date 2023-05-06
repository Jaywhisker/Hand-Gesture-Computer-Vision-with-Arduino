//importing dependencies
#include "Firebase_Arduino_WiFiNINA.h"

//define firebase and wifi
#define DATABASE_URL "" //your Firebase URL here 

//<databaseName>.firebaseio.com or <databaseName>.<region>.firebasedatabase.app
#define DATABASE_SECRET "" //your Firebase Database secret here

#define WIFI_SSID "" //your wifi name
#define WIFI_PASSWORD "" //your wifi password

//define Firebase data object
FirebaseData fbdo;

//define LED pin
int LED = 13;

void setup()
{
  Serial.begin(115200);
  delay(100);
  
  Serial.print("Connecting to Wi-Fi");
  int status = WL_IDLE_STATUS;
  while (status != WL_CONNECTED) //check if connected, if not try connecting
  {
    status = WiFi.begin(WIFI_SSID, WIFI_PASSWORD); //try connecting to wifi
    delay(100);
  }

  Serial.print("Connected to Wifi"); 

  //provide the autntication data
  Firebase.begin(DATABASE_URL, DATABASE_SECRET, WIFI_SSID, WIFI_PASSWORD);
  
  //if connection fail, reconnect if there is wifi access
	Firebase.reconnectWiFi(true);

  //setting LED as output
  pinMode(LED, OUTPUT);
}


//endlessly loop to read the data from firebase
void loop()
{
  //check that there is a led data in Firebase that is an int (0 or 1)
  if (Firebase.getInt(fbdo, "/led")) 
  {
    digitalWrite(LED, fbdo.intData()); //write to LED to on or off according to Firebase
  }

  else{
    Serial.println("error");
  }

  delay(500);  //pause for 0.5s
}
