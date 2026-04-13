import React from 'react';
import { View, Text, Button } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

const Streak_page = ({ onNavigateHome }) => {
  return (
    <SafeAreaProvider>
      <SafeAreaView>
        <View>
          <Text>Streak</Text>
        </View>
        <View>
          <Button title="Back" onPress={onNavigateHome} />
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
};
export default Streak_page;
