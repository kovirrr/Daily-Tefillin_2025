import React from 'react';
import { View, Text } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

function App() {
  return (
    <SafeAreaProvider>
      <SafeAreaView>
        <View>
          <Text>Daily Tefillin</Text>
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

export default App;
