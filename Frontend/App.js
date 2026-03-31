import { React, useState } from 'react';
import { View, Text, Button } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Faq_page from './components/faq_page/faq_page';

function App() {
  const pages = ['faq_page', 'main_page', 'steak_page'];
  const [curPage, setCurPage] = useState(pages[1]);
  if (curPage === pages[0]) {
    return <Faq_page onNavigateHome={setCurPage(pages[1])} />;
  } else if (curPage === pages[1]) {
    return (
      <SafeAreaProvider>
        <SafeAreaView>
          <View>
            <Text>Daily Tefillin</Text>
          </View>
          <View>
            <Button
              title="FAQ Page"
              onPress={setCurPage(pages[0])}
              color="red"
            />
          </View>
        </SafeAreaView>
      </SafeAreaProvider>
    );
  } else {
    return (
      <View>
        <Text>Page Not found</Text>
      </View>
    );
  }
}

export default App;
