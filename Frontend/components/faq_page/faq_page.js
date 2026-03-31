import React from 'react';
import { View, Text } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

const Faq_page = ({ onNavigateHome }) => {
  return (
    <SafeAreaProvider>
      <SafeAreaView>
        <View>
          <Text>FAQ</Text>
        </View>
        <View>
          <Button title="Back" onPress={onNavigateHome} />
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
};
export default Faq_page;
