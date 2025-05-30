import React from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  Linking, 
  Platform, 
  KeyboardAvoidingView 
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

const DemoPage = ({ navigation }) => {
  // Function to handle the press and navigate to a URL
  const handlePress = () => {
    const url = 'https://sunny-gecko-a6291c.netlify.app/'; // Replace with the desired URL
    Linking.openURL(url).catch(err => console.error('An error occurred', err));
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Text style={styles.title}>Demo Page</Text>

      <TouchableOpacity onPress={handlePress}>
        <Text style={styles.text}>Click here to go to the demo website</Text>
      </TouchableOpacity>

      {/* Footer for Navigation */}
      <View style={styles.footer}>
        <TouchableOpacity 
          style={styles.footerButton} 
          onPress={() => navigation.navigate('TranslateApp')} // Navigate to Home
        >
          <Icon name="home" size={24} color="#00CFFF" />
          <Text style={styles.footerButtonText}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.footerButton} 
          onPress={() => navigation.navigate('Settings')} // Navigate to Settings
        >
          <Icon name="settings" size={24} color="#00CFFF" />
          <Text style={styles.footerButtonText}>Settings</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',  // Same background color as the Settings page
    paddingHorizontal: 20,
    paddingTop: 50,
    justifyContent: 'center',  // Center the content vertically
    alignItems: 'center',
  },
  title: {
    fontWeight: 'bold',
    fontSize: 24,
    textAlign: 'center',
    color: '#fff',
    marginBottom: 20,
  },
  text: {
    fontSize: 18,
    color: '#00CFFF',
    marginTop: 20,
    textDecorationLine: 'underline',  // To make it look like a link
  },
  footer: {
    flexDirection: 'row', // Set to row so the buttons align horizontally
    justifyContent: 'space-around', // Evenly space out the buttons
    alignItems: 'center',
    paddingVertical: 10,
    backgroundColor: '#000',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 60, // Increased height to accommodate the buttons
  },
  footerButton: {
    alignItems: 'center',
  },
  footerButtonText: {
    color: '#00CFFF',
    fontSize: 14,
    marginTop: 4,
  },
});

export default DemoPage;
