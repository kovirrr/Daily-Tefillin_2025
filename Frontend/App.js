import { React, useState } from 'react';
import { View, Text, Button } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Faq_page from './components/faq_page/faq_page';
import Streak_page from './components/streak_page/streak_page';

function App() {
  const pages = ['faq_page', 'main_page', 'streak_page'];
  const [curPage, setCurPage] = useState(pages[1]);
  if (curPage === pages[0]) {
    return <Faq_page onNavigateHome={() => setCurPage(pages[1])} />;
  } else if (curPage === pages[2]) {
    return <Streak_page onNavigateHome={() => setCurPage(pages[1])} />;
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
              onPress={() => setCurPage(pages[0])}
              color="red"
            />
          </View>
          <View>
            <Button
              title="Streak Page"
              onPress={() => setCurPage(pages[2])}
              color="green"
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
